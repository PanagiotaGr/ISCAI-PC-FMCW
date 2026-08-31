from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path

ROOT = Path("/home/agni/waymo")
S5 = ROOT / "iscai_stage5"

CHECKER = (
    S5 / "scripts/"
    "run_stage5_fullpdf_v2_independent_formal_checker.py"
)

CHECKER2 = (
    S5 / "scripts/check_stage5_fullpdf_v2.py"
)

ART = (
    S5 / "artifacts/fullpdf_v2_independent_formal"
)

SEALED = (
    ART / "decision_ledger_sealed.jsonl"
)

V3_LOG = (
    ART
    / "checker2_superseding_receiver_binding_checkerfix_"
      "pretruthfix_v3.log"
)

OUTDIR = (
    ART / "truth_candidate_postseal_diagnostic_v1"
)

REPORT = (
    OUTDIR / "truth_candidate_postseal_diagnostic_v1.json"
)

EXPECTED = {
    CHECKER:
        "342344902e6b00cb841036dcdc0e54f1e1ba396344fa602864521890f5eb5da5",
    CHECKER2:
        "5e77d678af188d3f0b34b84ab53ddc78fc1b2921c6bea573dd9722c439c63599",
    SEALED:
        "751703975b3fc837cc1cf660cf1321db5768323d6cf2fbef524d05a3dbb48af4",
    V3_LOG:
        "56f3c3c6cbade9546b1792b7a79735f02e95f96a1f956ae9a6b5d927f38dbbc1",
}


def fail(msg: str) -> None:
    raise SystemExit("FAIL-CLOSED: " + msg)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(b)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for lineno, line in enumerate(
            f,
            1,
        ):
            text = line.strip()
            if not text:
                continue
            obj = json.loads(text)
            if not isinstance(obj, dict):
                fail(
                    f"non-object JSONL row "
                    f"{path}:{lineno}"
                )
            out.append(obj)
    return out


def canonical(obj) -> str:
    return (
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )


# ------------------------------------------------------------
# 1. Preserve exact V3 authority.
# ------------------------------------------------------------

for path, expected in EXPECTED.items():
    if not path.is_file():
        fail(
            f"missing protected V3 artifact: {path}"
        )

    actual = sha256(path)

    if actual != expected:
        fail(
            f"protected SHA mismatch: {path}\n"
            f"actual   = {actual}\n"
            f"expected = {expected}"
        )

mode = stat.S_IMODE(
    SEALED.stat().st_mode
)

if mode & 0o222:
    fail(
        "sealed decision ledger is writable"
    )

sealed_rows = load_jsonl(
    SEALED
)

if not sealed_rows:
    fail(
        "sealed decision ledger is empty"
    )

decision_keys = set()

for i, row in enumerate(
    sealed_rows,
    1,
):
    if (
        "scenario_id" not in row
        or "prediction_id" not in row
    ):
        fail(
            f"sealed row {i} lacks "
            "(scenario_id,prediction_id)"
        )

    decision_keys.add(
        (
            str(row["scenario_id"]),
            str(row["prediction_id"]),
        )
    )

# The formal receiver population is 120 scenes.
if len(decision_keys) != 120:
    fail(
        "unexpected sealed unique receiver-key count: "
        f"{len(decision_keys)}"
    )

# ------------------------------------------------------------
# 2. Import checker as library only.
#    __main__ is NOT executed.
# ------------------------------------------------------------

spec = importlib.util.spec_from_file_location(
    "stage5_formal_checker_postseal_diag",
    CHECKER,
)

if (
    spec is None
    or spec.loader is None
):
    fail(
        "could not create checker import spec"
    )

mod = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    mod
)

for name in (
    "candidate_v2_paths",
    "resolve_truth_binding",
):
    if not hasattr(
        mod,
        name,
    ):
        fail(
            f"checker lacks {name}"
        )

original_candidate_fn = (
    mod.candidate_v2_paths
)

# ------------------------------------------------------------
# 3. Discover exactly what the frozen checker considers
#    truth candidates.
#
#    This is POST-SEAL evaluator access.
# ------------------------------------------------------------

try:
    candidates = list(
        original_candidate_fn(
            truth_phase=True
        )
    )
except TypeError:
    # Keep compatibility if positional-only.
    candidates = list(
        original_candidate_fn(
            True
        )
    )

candidate_paths = []

for value in candidates:
    p = Path(value)

    if not p.is_file():
        continue

    if p not in candidate_paths:
        candidate_paths.append(
            p
        )

candidate_paths.sort(
    key=lambda p: str(p)
)

# ------------------------------------------------------------
# 4. Test EACH candidate in isolation using the unchanged
#    resolve_truth_binding() itself.
#
#    We do NOT reimplement its semantics.
#    We monkeypatch only candidate discovery in this temporary
#    diagnostic process, so the resolver sees one candidate.
# ------------------------------------------------------------

results = []

for path in candidate_paths:
    result = {
        "path": str(path),
        "name": path.name,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "single_candidate_resolution":
            "NOT_RUN",
        "resolved_truth_rows": None,
        "exception_type": None,
        "exception_message": None,
    }

    # Fresh process-local logical state for each isolated test.
    #
    # The physical ledger is independently verified above as
    # immutable/read-only. Setting this global merely tells the
    # imported resolver that the already-completed V3 seal exists.
    if hasattr(
        mod,
        "DECISION_LEDGER_SEALED",
    ):
        mod.DECISION_LEDGER_SEALED = True

    if hasattr(
        mod,
        "FORMAL_TRUTH_OPENED",
    ):
        mod.FORMAL_TRUTH_OPENED = False

    if hasattr(
        mod,
        "EVENTS",
    ):
        try:
            mod.EVENTS.clear()
        except Exception:
            pass

    def only_this_candidate(
        *args,
        _path=path,
        **kwargs,
    ):
        return [_path]

    mod.candidate_v2_paths = (
        only_this_candidate
    )

    try:
        resolved_path, truth_map = (
            mod.resolve_truth_binding(
                decision_keys
            )
        )

        result[
            "single_candidate_resolution"
        ] = "PASS"

        result[
            "resolved_path"
        ] = str(
            resolved_path
        )

        try:
            result[
                "resolved_truth_rows"
            ] = len(
                truth_map
            )
        except Exception:
            result[
                "resolved_truth_rows"
            ] = None

    except BaseException as exc:
        result[
            "single_candidate_resolution"
        ] = "FAIL"

        result[
            "exception_type"
        ] = type(exc).__name__

        # No truth values are emitted; only resolver diagnostic.
        msg = str(exc).replace(
            "\n",
            " ",
        )

        result[
            "exception_message"
        ] = msg[:500]

    results.append(
        result
    )

# Restore checker function even though process ends.
mod.candidate_v2_paths = (
    original_candidate_fn
)

# ------------------------------------------------------------
# 5. Distinct valid bindings.
# ------------------------------------------------------------

valid = [
    r
    for r in results
    if (
        r[
            "single_candidate_resolution"
        ]
        == "PASS"
    )
]

valid_sha = sorted(
    {
        r["sha256"]
        for r in valid
    }
)

report = {
    "stage": 5,
    "version": "fullpdf_v2",
    "purpose":
        "post-seal evaluator-truth candidate resolver diagnostic",
    "scientific_boundary": {
        "decision_ledger_recomputed":
            False,
        "controller_reexecuted":
            False,
        "beam_decisions_changed":
            False,
        "formal_metrics_computed":
            False,
        "formal_acceptance_evaluated":
            False,
        "evaluator_truth_accessed":
            True,
        "evaluator_truth_access":
            "after immutable V3 decision seal only",
        "post_outcome_tuning":
            False,
    },
    "protected_V3": {
        "sealed_decision_ledger":
            str(SEALED),
        "sealed_decision_ledger_sha256":
            sha256(SEALED),
        "sealed_read_only":
            True,
        "sealed_rows":
            len(sealed_rows),
        "unique_decision_keys":
            len(decision_keys),
        "checker_sha256":
            sha256(CHECKER),
        "checker2_sha256":
            sha256(CHECKER2),
        "V3_log_sha256":
            sha256(V3_LOG),
    },
    "candidate_count":
        len(candidate_paths),
    "single_candidate_results":
        results,
    "valid_single_candidate_count":
        len(valid),
    "valid_distinct_sha_count":
        len(valid_sha),
    "valid_distinct_sha256":
        valid_sha,
    "status":
        (
            "EXACTLY_ONE_VALID_DISTINCT_BINDING"
            if len(valid_sha) == 1
            else
            "MULTIPLE_VALID_DISTINCT_BINDINGS"
            if len(valid_sha) > 1
            else
            "NO_EXISTING_VALID_BINDING"
        ),
    "Stage6_allowed":
        False,
}

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

if REPORT.exists():
    fail(
        "diagnostic V1 report already exists"
    )

REPORT.write_text(
    canonical(report),
    encoding="utf-8",
)

os.chmod(
    REPORT,
    0o444,
)

print("=" * 78)
print(
    "STAGE5 POST-SEAL TRUTH-CANDIDATE DIAGNOSTIC"
)
print("=" * 78)

print(
    "sealed_sha256 =",
    sha256(SEALED),
)

print(
    "sealed_rows =",
    len(sealed_rows),
)

print(
    "unique_decision_keys =",
    len(decision_keys),
)

print(
    "truth_candidates =",
    len(candidate_paths),
)

print()
print(
    "===== ONE LINE PER CHECKER-DISCOVERED CANDIDATE ====="
)

for i, r in enumerate(
    results,
    1,
):
    status = r[
        "single_candidate_resolution"
    ]

    short_sha = r[
        "sha256"
    ][:12]

    if status == "PASS":
        detail = (
            "rows="
            + str(
                r[
                    "resolved_truth_rows"
                ]
            )
        )
    else:
        detail = (
            str(
                r[
                    "exception_type"
                ]
            )
            + ": "
            + str(
                r[
                    "exception_message"
                ]
            )[:180]
        )

    print(
        f"{i:02d} | {status:4s} | "
        f"{short_sha} | "
        f"{r['name']} | "
        f"{detail}"
    )

print()
print(
    "valid_single_candidate_count =",
    len(valid),
)

print(
    "valid_distinct_sha_count =",
    len(valid_sha),
)

print(
    "diagnostic_status =",
    report["status"],
)

print(
    "formal_metrics_computed = false"
)

print(
    "controller_reexecuted = false"
)

print(
    "evaluator_truth_accessed = true "
    "(POST-SEAL ONLY)"
)

print(
    "Stage6_allowed = false"
)

print(
    "report =",
    REPORT,
)

print(
    "report_sha256 =",
    sha256(REPORT),
)

print("=" * 78)

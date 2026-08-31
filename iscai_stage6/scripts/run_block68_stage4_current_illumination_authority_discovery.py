from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback


S6 = Path("/home/agni/waymo/iscai_stage6")

PREREG = (
    S6
    / "configs/"
      "block66_part3b_class_aware_policy_preregistration.json"
)

STAGE3_FREEZE = (
    S6
    / "configs/"
      "stage6_development_stage3_class_floor_freeze.json"
)

STAGE3_SCORING = (
    S6
    / "configs/"
      "stage6_development_stage3_floor_scoring_semantics.json"
)

REACTIVE_SCORING = (
    S6
    / "configs/"
      "stage6_reactive_development_scoring_contract.json"
)

REACTIVE_REPORT = (
    S6
    / "reports/"
      "block68_reactive_decision_ledger_freeze.json"
)

CLASS_RUNTIME = (
    S6
    / "src/iscai_stage6/adb/"
      "class_aware_policy.py"
)

REACTIVE_RUNTIME = (
    S6
    / "src/iscai_stage6/adb/"
      "reactive_development_scoring.py"
)

REPORT = (
    S6
    / "reports/"
      "block68_stage4_current_illumination_authority_discovery.json"
)

EXPECTED = {
    PREREG:
        "5704494a89b4b3c7a3c19b0df8176c550c00795ed17cf2906c343f340461429e",

    STAGE3_FREEZE:
        "c8536a1a2f888ebc3310cb900626cf0c8c0c2ca3ce34c2d321120a269c548d00",

    REACTIVE_SCORING:
        "454d22a2f067025fd8dde877546b7305c4357000126585c860a4a2d9ef3535d5",
}

EXPECTED_TESTS = 224

SEARCH_TERMS = (
    "current_illumination",
    "current causal illumination",
    "current_causal_illumination",
    "t0",
    "reactive map",
    "reactive_map",
    "decision map",
    "decision_map",
    "illumination_path",
    "map_path",
    "hold_t0",
    "all-on",
    "all_on",
)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def file_sha(path: Path) -> str:
    h = sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def canonical_bytes(value):
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def is_actual_formal_path(path: Path) -> bool:
    text = str(path).lower()

    # "preformal" is explicitly allowed.
    text = text.replace(
        "preformal",
        "",
    )

    return "formal" in text


def safe_text(path: Path):
    if not path.is_file():
        return None

    if is_actual_formal_path(path):
        return None

    try:
        return path.read_text(
            encoding="utf-8",
        )
    except Exception:
        return None


def flatten(value, prefix=""):
    rows = []

    if isinstance(value, dict):
        for key, item in value.items():
            name = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            rows.extend(
                flatten(
                    item,
                    name,
                )
            )

    elif isinstance(value, list):
        for index, item in enumerate(value):
            name = (
                f"{prefix}[{index}]"
            )

            rows.extend(
                flatten(
                    item,
                    name,
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


def context_hits(path: Path, text: str):
    lines = text.splitlines()
    hits = []

    for index, line in enumerate(lines):
        lower = line.lower()

        matched = [
            term
            for term in SEARCH_TERMS
            if term.lower() in lower
        ]

        if not matched:
            continue

        lo = max(
            0,
            index - 3,
        )

        hi = min(
            len(lines),
            index + 4,
        )

        hits.append(
            {
                "line":
                    index + 1,

                "matched_terms":
                    matched,

                "context":
                    [
                        f"L{j + 1:05d}: {lines[j]}"
                        for j in range(
                            lo,
                            hi,
                        )
                    ],
            }
        )

    return hits


def inspect_json(path: Path):
    text = safe_text(path)

    if text is None:
        return None

    try:
        value = json.loads(text)
    except Exception:
        return None

    leaves = flatten(value)

    relevant = []

    for key, value in leaves:
        haystack = (
            str(key)
            + " "
            + str(value)
        ).lower()

        if any(
            term.lower() in haystack
            for term in SEARCH_TERMS
        ):
            relevant.append(
                {
                    "key":
                        key,
                    "value":
                        value,
                }
            )

    return {
        "sha256":
            file_sha(path),

        "relevant_leaves":
            relevant,
    }


def inspect_jsonl_metadata(path: Path):
    if not path.is_file():
        return None

    if is_actual_formal_path(path):
        return None

    row_count = 0
    key_paths = set()
    relevant_leaves = []
    path_like_values = set()

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:

        for line_number, raw in enumerate(
            stream,
            start=1,
        ):
            line = raw.strip()

            if not line:
                continue

            value = json.loads(line)

            row_count += 1

            leaves = flatten(value)

            for key, item in leaves:
                key_paths.add(key)

                haystack = (
                    str(key)
                    + " "
                    + str(item)
                ).lower()

                if any(
                    term.lower() in haystack
                    for term in SEARCH_TERMS
                ):
                    if len(relevant_leaves) < 100:
                        relevant_leaves.append(
                            {
                                "row":
                                    row_count,

                                "key":
                                    key,

                                "value":
                                    item,
                            }
                        )

                if isinstance(item, str):
                    candidate = item.lower()

                    if (
                        "/" in item
                        or candidate.endswith(
                            (
                                ".npy",
                                ".npz",
                                ".json",
                                ".jsonl",
                            )
                        )
                    ):
                        path_like_values.add(
                            item
                        )

    return {
        "row_count":
            row_count,

        "sha256":
            file_sha(path),

        "key_paths":
            sorted(
                key_paths
            ),

        "relevant_leaves":
            relevant_leaves,

        "path_like_values":
            sorted(
                path_like_values
            ),
    }


def inventory_candidate_files():
    roots = (
        S6 / "artifacts/block68",
        S6 / "configs",
        S6 / "reports",
    )

    tokens = (
        "reactive",
        "decision",
        "t0",
        "illum",
        "map",
        "current",
    )

    found = []

    for root in roots:
        if not root.is_dir():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if is_actual_formal_path(path):
                continue

            name = path.name.lower()

            if not any(
                token in name
                for token in tokens
            ):
                continue

            found.append(
                {
                    "path":
                        str(path),

                    "bytes":
                        path.stat().st_size,

                    "sha256":
                        file_sha(path),
                }
            )

    return sorted(
        found,
        key=lambda row: row["path"],
    )


def find_reactive_ledger_candidates():
    candidates = []

    roots = (
        S6 / "artifacts/block68",
        S6 / "reports",
    )

    for root in roots:
        if not root.is_dir():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if is_actual_formal_path(path):
                continue

            low = path.name.lower()

            if (
                "reactive" in low
                and
                (
                    "ledger" in low
                    or
                    "decision" in low
                    or
                    "map" in low
                )
            ):
                candidates.append(
                    path
                )

    return sorted(
        set(candidates),
        key=str,
    )


def run_regression():
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
        ],
        cwd=str(S6),
        env=dict(
            **__import__("os").environ
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    count = None

    for line in process.stdout.splitlines():
        match = re.search(
            r"Ran\s+(\d+)\s+tests?",
            line,
        )

        if match:
            count = int(
                match.group(1)
            )

    require(
        process.returncode == 0,
        (
            "Stage6 regression failed:\n"
            + "\n".join(
                process.stdout.splitlines()[-100:]
            )
        ),
    )

    require(
        count == EXPECTED_TESTS,
        (
            f"Expected {EXPECTED_TESTS} tests; "
            f"observed {count}."
        ),
    )

    return count


def main():

    print("=" * 78)
    print("STAGE 6 — BLOCK 6.8")
    print("STAGE-4 CURRENT-ILLUMINATION AUTHORITY DISCOVERY")
    print("READ ONLY — NO TEMPORAL/RATE OUTCOMES")
    print("=" * 78)

    # --------------------------------------------------------
    # A. Frozen boundary
    # --------------------------------------------------------

    print()
    print("===== A. EXACT FROZEN BOUNDARY =====")

    for path, expected_sha in EXPECTED.items():
        require(
            path.is_file(),
            f"Missing required artifact: {path}",
        )

        actual = file_sha(path)

        require(
            actual == expected_sha,
            (
                f"SHA mismatch for {path}\n"
                f"expected={expected_sha}\n"
                f"actual  ={actual}"
            ),
        )

        print(
            path.name,
            "= EXACT PASS",
        )

    require(
        CLASS_RUNTIME.is_file(),
        f"Missing {CLASS_RUNTIME}",
    )

    print(
        "Stage-1 gamma          = FROZEN"
    )

    print(
        "Stage-2 margins        = FROZEN"
    )

    print(
        "Stage-3 floors         = FROZEN"
    )

    print(
        "Stage-4 outcomes       = NOT COMPUTED"
    )

    print(
        "Stage-4 winner         = NOT SELECTED"
    )

    print(
        "primary acceptance     = NOT TESTED"
    )

    print(
        "formal evaluation      = NO"
    )

    # --------------------------------------------------------
    # B. Stage-4 preregistration
    # --------------------------------------------------------

    print()
    print("===== B. PREREGISTERED STAGE-4 CONTRACT =====")

    prereg = json.loads(
        PREREG.read_text(
            encoding="utf-8"
        )
    )

    stage4 = (
        prereg[
            "development_selection_protocol"
        ][
            "stage_4_temporal_and_rate"
        ]
    )

    print(
        json.dumps(
            stage4,
            indent=2,
            sort_keys=True,
        )
    )

    temporal = prereg.get(
        "temporal_smoothing",
        {},
    )

    rate = prereg.get(
        "actuation_rate_limit",
        {},
    )

    print()
    print("temporal_smoothing:")
    print(
        json.dumps(
            temporal,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print("actuation_rate_limit:")
    print(
        json.dumps(
            rate,
            indent=2,
            sort_keys=True,
        )
    )

    # --------------------------------------------------------
    # C. Runtime authority
    # --------------------------------------------------------

    print()
    print("===== C. CURRENT-STATE RUNTIME SEMANTICS =====")

    runtime_text = safe_text(
        CLASS_RUNTIME
    )

    require(
        runtime_text is not None,
        "Could not read class-aware runtime.",
    )

    runtime_hits = context_hits(
        CLASS_RUNTIME,
        runtime_text,
    )

    for item in runtime_hits:
        if (
            "current_illumination"
            not in
            " ".join(
                item["matched_terms"]
            ).lower()
            and
            "current causal illumination"
            not in
            " ".join(
                item["matched_terms"]
            ).lower()
        ):
            continue

        print()
        for line in item["context"]:
            print(line)

    require(
        "Initial I_prev is current causal illumination."
        in
        runtime_text,
        (
            "Frozen temporal runtime no longer contains "
            "the current-causal-illumination contract."
        ),
    )

    print()
    print(
        "runtime initial-state contract = "
        "CURRENT CAUSAL ILLUMINATION"
    )

    # --------------------------------------------------------
    # D. Prove smoke all-on is not selection authority
    # --------------------------------------------------------

    print()
    print("===== D. STRUCTURAL-SMOKE ALL-ON STATUS =====")

    smoke_sources = []

    for path in (
        S6 / "scripts"
    ).glob(
        "run_block66_part3c*.py"
    ):
        text = safe_text(path)

        if text is None:
            continue

        if (
            "Neutral causal all-on state"
            in text
            or
            "operator-chain plumbing smoke"
            in text
        ):
            smoke_sources.append(
                {
                    "path":
                        str(path),

                    "sha256":
                        file_sha(path),

                    "hits":
                        context_hits(
                            path,
                            text,
                        ),
                }
            )

            print()
            print(
                path,
                "SHA256=",
                file_sha(path),
            )

            lines = text.splitlines()

            for index, line in enumerate(lines):
                if (
                    "Neutral causal all-on state"
                    in line
                    or
                    "operator-chain plumbing smoke"
                    in line
                ):
                    lo = max(
                        0,
                        index - 2,
                    )

                    hi = min(
                        len(lines),
                        index + 8,
                    )

                    for j in range(
                        lo,
                        hi,
                    ):
                        print(
                            f"L{j + 1:05d}: "
                            f"{lines[j]}"
                        )

    require(
        smoke_sources,
        (
            "Could not recover structural-smoke "
            "all-on disclaimer."
        ),
    )

    print()
    print(
        "all-on smoke state usable for Stage-4 selection = NO"
    )

    # --------------------------------------------------------
    # E. Existing frozen reactive/current-state contracts
    # --------------------------------------------------------

    print()
    print("===== E. FROZEN REACTIVE/CURRENT-STATE CONTRACTS =====")

    json_sources = {}

    for path in (
        REACTIVE_SCORING,
        REACTIVE_REPORT,
        STAGE3_SCORING,
    ):
        if not path.is_file():
            print(
                path,
                "= MISSING"
            )
            continue

        inspected = inspect_json(
            path
        )

        json_sources[
            str(path)
        ] = inspected

        print()
        print(
            path,
            "SHA256=",
            file_sha(path),
        )

        if inspected:
            for row in inspected[
                "relevant_leaves"
            ]:
                print(
                    f"  {row['key']} = "
                    f"{row['value']!r}"
                )

    if REACTIVE_RUNTIME.is_file():
        print()
        print(
            REACTIVE_RUNTIME,
            "SHA256=",
            file_sha(REACTIVE_RUNTIME),
        )

        text = safe_text(
            REACTIVE_RUNTIME
        )

        if text is not None:
            hits = context_hits(
                REACTIVE_RUNTIME,
                text,
            )

            for item in hits:
                print()
                for line in item["context"]:
                    print(line)

    # --------------------------------------------------------
    # F. Discover exact decision-ledger/map artifacts
    # --------------------------------------------------------

    print()
    print("===== F. REACTIVE DECISION / MAP ARTIFACT DISCOVERY =====")

    ledger_candidates = (
        find_reactive_ledger_candidates()
    )

    print(
        "candidate files =",
        len(ledger_candidates),
    )

    ledger_details = {}

    for path in ledger_candidates:
        print()
        print(
            path,
            "| bytes=",
            path.stat().st_size,
            "| sha256=",
            file_sha(path),
        )

        if path.suffix == ".json":
            value = inspect_json(
                path
            )

            ledger_details[
                str(path)
            ] = value

            if value:
                for row in value[
                    "relevant_leaves"
                ]:
                    print(
                        f"  {row['key']} = "
                        f"{row['value']!r}"
                    )

        elif path.suffix == ".jsonl":
            value = inspect_jsonl_metadata(
                path
            )

            ledger_details[
                str(path)
            ] = value

            if value:
                print(
                    "  rows =",
                    value[
                        "row_count"
                    ],
                )

                print(
                    "  relevant key paths:"
                )

                for key in value[
                    "key_paths"
                ]:
                    lower = key.lower()

                    if any(
                        token in lower
                        for token in (
                            "map",
                            "illum",
                            "decision",
                            "reactive",
                            "hold",
                        )
                    ):
                        print(
                            "   ",
                            key,
                        )

                if value[
                    "relevant_leaves"
                ]:
                    print(
                        "  relevant leaves:"
                    )

                    for row in value[
                        "relevant_leaves"
                    ][:30]:
                        print(
                            "   ",
                            f"row={row['row']}",
                            f"{row['key']}="
                            f"{row['value']!r}",
                        )

                if value[
                    "path_like_values"
                ]:
                    print(
                        "  path-like values:"
                    )

                    for item in value[
                        "path_like_values"
                    ]:
                        print(
                            "   ",
                            item,
                        )

    # --------------------------------------------------------
    # G. Broader filename-only inventory
    # --------------------------------------------------------

    print()
    print("===== G. CURRENT/REACTIVE/MAP FILE INVENTORY =====")

    inventory = (
        inventory_candidate_files()
    )

    for row in inventory:
        print(
            row["path"],
            "| bytes=",
            row["bytes"],
            "| sha256=",
            row["sha256"],
        )

    # --------------------------------------------------------
    # H. Source-only trace
    # --------------------------------------------------------

    print()
    print("===== H. SOURCE TRACE FOR T0/CURRENT MAP PRODUCTION =====")

    source_candidates = []

    for root in (
        S6 / "scripts",
        S6 / "src/iscai_stage6",
    ):
        if not root.is_dir():
            continue

        for path in root.rglob("*.py"):
            if is_actual_formal_path(
                path
            ):
                continue

            text = safe_text(
                path
            )

            if text is None:
                continue

            lower = text.lower()

            score = sum(
                int(term in lower)
                for term in (
                    "reactive",
                    "current_illumination",
                    "decision",
                    "t0",
                    "illumination_map",
                )
            )

            if score < 2:
                continue

            hits = context_hits(
                path,
                text,
            )

            if not hits:
                continue

            source_candidates.append(
                {
                    "path":
                        str(path),

                    "sha256":
                        file_sha(path),

                    "score":
                        score,

                    "hits":
                        hits,
                }
            )

    source_candidates.sort(
        key=lambda row: (
            -row["score"],
            row["path"],
        )
    )

    for row in source_candidates[:20]:
        print()
        print(
            row["path"],
            "| score=",
            row["score"],
            "| sha256=",
            row["sha256"],
        )

        printed = 0

        for hit in row["hits"]:
            interesting = (
                " ".join(
                    hit[
                        "matched_terms"
                    ]
                ).lower()
            )

            if not any(
                token in interesting
                for token in (
                    "current_illumination",
                    "t0",
                    "reactive map",
                    "decision map",
                    "map_path",
                    "illumination_path",
                    "hold_t0",
                )
            ):
                continue

            print()

            for line in hit["context"]:
                print(line)

            printed += 1

            if printed >= 5:
                break

    # --------------------------------------------------------
    # I. Classification
    # --------------------------------------------------------

    print()
    print("===== I. AUTHORITY CLASSIFICATION =====")

    explicit_authority_hits = []

    for source in (
        runtime_text,
        safe_text(REACTIVE_SCORING) or "",
        safe_text(REACTIVE_RUNTIME) or "",
    ):
        low = source.lower()

        phrases = (
            "current causal illumination",
            "current_illumination",
            "hold t0",
            "hold_t0",
        )

        for phrase in phrases:
            if phrase in low:
                explicit_authority_hits.append(
                    phrase
                )

    print(
        "runtime requires current causal state = YES"
    )

    print(
        "structural all-on is selection authority = NO"
    )

    print(
        "reactive/map candidates discovered =",
        len(
            ledger_candidates
        ),
    )

    print(
        "explicit current-state text hits =",
        sorted(
            set(
                explicit_authority_hits
            )
        ),
    )

    print()
    print(
        "CLASSIFICATION = "
        "CURRENT_STATE_SOURCE_MUST_BE_BOUND_BEFORE_STAGE4_SWEEP"
    )

    print(
        "No Stage-4 tuple may be scored until the exact "
        "2-D I_prev source is identified and frozen."
    )

    # --------------------------------------------------------
    # J. Regression
    # --------------------------------------------------------

    print()
    print("===== J. FULL STAGE6 REGRESSION =====")

    tests = run_regression()

    print(
        "Stage6 regression =",
        f"{tests} / {tests} PASS",
    )

    # --------------------------------------------------------
    # K. Report
    # --------------------------------------------------------

    report = {
        "stage":
            6,

        "block":
            "6.8_stage4_current_illumination_authority_discovery",

        "status":
            "PASS_STAGE4_CURRENT_ILLUMINATION_AUTHORITY_DISCOVERY_ONLY",

        "scientific_boundary": {
            "Stage1_gamma":
                "FROZEN",

            "Stage2_margins":
                "FROZEN",

            "Stage3_floors":
                "FROZEN",

            "Stage4_outcomes_computed":
                False,

            "Stage4_winner_selected":
                False,

            "primary_acceptance_tested":
                False,

            "formal_evaluation":
                False,

            "policy_numerics_modified":
                False,
        },

        "authority": {
            "runtime_initial_state":
                "current causal illumination",

            "structural_smoke_all_on_authoritative_for_selection":
                False,

            "current_state_binding_required_before_sweep":
                True,
        },

        "frozen_inputs": {
            str(path):
                file_sha(path)
            for path in EXPECTED
        },

        "reactive_decision_candidates":
            [
                str(path)
                for path in ledger_candidates
            ],

        "inventory":
            inventory,

        "source_candidates":
            source_candidates,

        "json_sources":
            json_sources,

        "ledger_details":
            ledger_details,

        "Stage6_regression":
            tests,

        "next":
            (
                "Bind the exact frozen 2-D current causal "
                "illumination I_prev source, then freeze "
                "Stage-4 scoring/initial-state semantics "
                "before any temporal/rate candidate outcome."
            ),
    }

    REPORT.write_bytes(
        canonical_bytes(
            report
        )
    )

    print()
    print("=" * 78)
    print("STAGE-4 CURRENT-ILLUMINATION DISCOVERY — FINAL")
    print("=" * 78)

    print(
        "Stage-1 gamma         = FROZEN"
    )

    print(
        "Stage-2 margins       = FROZEN"
    )

    print(
        "Stage-3 floors        = FROZEN"
    )

    print(
        "runtime I_prev        = CURRENT CAUSAL ILLUMINATION"
    )

    print(
        "all-on smoke state    = NOT SELECTION AUTHORITY"
    )

    print(
        "Stage-4 outcomes      = NOT COMPUTED"
    )

    print(
        "Stage-4 winner        = NOT SELECTED"
    )

    print(
        "primary acceptance    = NOT TESTED"
    )

    print(
        "formal evaluation     = NO"
    )

    print(
        "Stage6 regression     =",
        f"{tests} / {tests} PASS",
    )

    print(
        "STATUS = "
        "PASS_STAGE4_CURRENT_ILLUMINATION_AUTHORITY_DISCOVERY_ONLY"
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "terminal remains open = YES"
    )


try:
    main()

except BaseException as exc:

    print()
    print("=" * 78)
    print(
        "STAGE-4 CURRENT-ILLUMINATION DISCOVERY = BLOCKED"
    )
    print("=" * 78)

    print(
        "exception =",
        type(exc).__name__,
        str(exc),
    )

    traceback.print_exc()

    print()
    print(
        "Stage-1 gamma       = STILL FROZEN"
    )

    print(
        "Stage-2 margins     = STILL FROZEN"
    )

    print(
        "Stage-3 floors      = STILL FROZEN"
    )

    print(
        "Stage-4 sweep       = NOT ALLOWED"
    )

    print(
        "primary acceptance  = NOT TESTED"
    )

    print(
        "formal evaluation   = NO"
    )

    print()
    print(
        "Do not choose an all-on initial state and do "
        "not run the temporal/rate sweep."
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

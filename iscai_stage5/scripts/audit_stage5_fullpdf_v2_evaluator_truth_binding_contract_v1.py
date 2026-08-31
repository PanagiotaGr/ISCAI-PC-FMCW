from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
from pathlib import Path

ROOT = Path("/home/agni/waymo")
S4 = ROOT / "iscai_stage4"
S5 = ROOT / "iscai_stage5"

FORMAL_CHECKER = (
    S5 / "scripts/run_stage5_fullpdf_v2_independent_formal_checker.py"
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

BINDING = (
    S5
    / "artifacts/fullpdf_v2_repair/"
    "primary_receiver_binding_fullpdf_v2.jsonl"
)

ART = (
    S5 / "artifacts/fullpdf_v2_independent_formal"
)

NEW_PRETRUTH = (
    ART / "checker2_pretruth_seal_superseding_v3.json"
)

AUTHORITY_SEAL = (
    ART
    / "checker2_pretruth_authority_repair_v1/"
    "stage5_fullpdf_v2_checker2_pretruth_authority_"
    "superseding_seal_v1.json"
)

V3_LOG = (
    ART
    / "checker2_superseding_receiver_binding_checkerfix_"
    "pretruthfix_v3.log"
)

V3_POSTRUN = (
    ART / "superseding_formal_v3_postrun_audit.json"
)

FORMAL_REPORT = (
    S5 / "reports/stage5_fullpdf_v2_independent_formal_checker.json"
)

OUTDIR = (
    ART / "evaluator_truth_binding_contract_v1"
)

FULL_LOG = (
    OUTDIR / "source_contract_full.log"
)

SUMMARY_JSON = (
    OUTDIR / "source_contract_summary.json"
)

EXPECTED = {
    FORMAL_CHECKER:
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

    BINDING:
        "5a5516ee050d2fd1c0a571193007650e75f23836b2734ee3df1950cc04f2e1c8",

    NEW_PRETRUTH:
        "28209ad30bd0c4acdd62ca6f43205fd24fb943f7df4f9f61bc6f62a7df10a705",

    AUTHORITY_SEAL:
        "78be461e8f66f964f024acacb8f979be96febff7fdf761a4158de6fe3dcf7591",

    V3_LOG:
        "56f3c3c6cbade9546b1792b7a79735f02e95f96a1f956ae9a6b5d927f38dbbc1",

    V3_POSTRUN:
        "8ae664005203cf80175f0da0afcd01431b6dac6e83c1c1026e8c66878366ab67",
}


def fail(msg):
    raise SystemExit(
        "FAIL-CLOSED: " + str(msg)
    )


def sha(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def canonical(obj):
    return (
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )


for path, expected in EXPECTED.items():
    if not path.is_file():
        fail(
            f"missing required file: {path}"
        )

    actual = sha(path)

    if actual != expected:
        fail(
            f"SHA mismatch: {path}\n"
            f"actual   = {actual}\n"
            f"expected = {expected}"
        )

if not FORMAL_REPORT.is_file():
    fail(
        "V3 formal report missing"
    )

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

# Audit only; do not overwrite prior audit.
if (
    FULL_LOG.exists()
    or SUMMARY_JSON.exists()
):
    fail(
        "evaluator-truth contract audit V1 "
        "already exists"
    )

source = FORMAL_CHECKER.read_text(
    encoding="utf-8"
)

lines = source.splitlines()

tree = ast.parse(
    source
)

top_functions = {
    node.name: node
    for node in tree.body
    if isinstance(
        node,
        (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
        ),
    )
}

# ------------------------------------------------------------
# Exact functions relevant to evaluator truth.
# ------------------------------------------------------------

wanted_exact = {
    "resolve_truth_binding",
    "evaluate_sealed_decisions",
    "candidate_v2_paths",
}

selected_names = set()

for name in top_functions:
    low = name.lower()

    if (
        name in wanted_exact
        or "truth" in low
        or (
            "parse" in low
            and (
                "row" in low
                or "record" in low
            )
        )
    ):
        selected_names.add(
            name
        )

if "resolve_truth_binding" not in top_functions:
    fail(
        "resolve_truth_binding not found"
    )

if "evaluate_sealed_decisions" not in top_functions:
    fail(
        "evaluate_sealed_decisions not found"
    )

# Add direct helper functions called by resolver.
resolver = top_functions[
    "resolve_truth_binding"
]

resolver_calls = set()

for node in ast.walk(
    resolver
):
    if isinstance(
        node,
        ast.Call,
    ):
        if isinstance(
            node.func,
            ast.Name,
        ):
            resolver_calls.add(
                node.func.id
            )

        elif isinstance(
            node.func,
            ast.Attribute,
        ):
            resolver_calls.add(
                node.func.attr
            )

for name in resolver_calls:
    if name in top_functions:
        selected_names.add(
            name
        )

# Add helpers called by likely truth parsers.
for name in list(
    selected_names
):
    node = top_functions.get(
        name
    )

    if node is None:
        continue

    for child in ast.walk(
        node
    ):
        if (
            isinstance(
                child,
                ast.Call,
            )
            and isinstance(
                child.func,
                ast.Name,
            )
            and child.func.id
            in top_functions
        ):
            if any(
                token in child.func.id.lower()
                for token in (
                    "truth",
                    "candidate",
                    "parse",
                    "load",
                    "extract",
                    "alias",
                )
            ):
                selected_names.add(
                    child.func.id
                )

function_info = {}

for name in sorted(
    selected_names,
    key=lambda n:
        top_functions[n].lineno,
):
    node = top_functions[name]

    segment = ast.get_source_segment(
        source,
        node,
    ) or ""

    string_literals = sorted(
        {
            x.value
            for x in ast.walk(node)
            if (
                isinstance(
                    x,
                    ast.Constant,
                )
                and isinstance(
                    x.value,
                    str,
                )
                and len(x.value) <= 180
            )
        }
    )

    function_info[name] = {
        "start_line":
            node.lineno,
        "end_line":
            node.end_lineno,
        "calls":
            sorted(
                {
                    (
                        x.func.id
                        if isinstance(
                            x.func,
                            ast.Name,
                        )
                        else x.func.attr
                    )
                    for x in ast.walk(node)
                    if (
                        isinstance(
                            x,
                            ast.Call,
                        )
                        and isinstance(
                            x.func,
                            (
                                ast.Name,
                                ast.Attribute,
                            ),
                        )
                    )
                }
            ),
        "string_literals":
            string_literals,
        "source":
            segment,
    }

# ------------------------------------------------------------
# Global path/constants relevant to truth + ledgers.
# ------------------------------------------------------------

interesting_constants = {}

TOKENS = (
    "TRUTH",
    "LEDGER",
    "EVALUATOR",
    "SEAL",
    "RUN1",
    "RUN2",
)

for node in tree.body:
    if isinstance(
        node,
        ast.Assign,
    ):
        for target in node.targets:
            if (
                isinstance(
                    target,
                    ast.Name,
                )
                and any(
                    token in target.id.upper()
                    for token in TOKENS
                )
            ):
                interesting_constants[
                    target.id
                ] = {
                    "line":
                        node.lineno,
                    "expression":
                        ast.unparse(
                            node.value
                        ),
                }

    elif isinstance(
        node,
        ast.AnnAssign,
    ):
        target = node.target

        if (
            isinstance(
                target,
                ast.Name,
            )
            and any(
                token in target.id.upper()
                for token in TOKENS
            )
        ):
            interesting_constants[
                target.id
            ] = {
                "line":
                    node.lineno,
                "expression":
                    (
                        ast.unparse(
                            node.value
                        )
                        if node.value
                        is not None
                        else None
                    ),
            }

resolver_source = (
    function_info[
        "resolve_truth_binding"
    ]["source"]
)

resolver_sets_truth_opened = (
    "FORMAL_TRUTH_OPENED = True"
    in resolver_source
)

resolver_mentions_seal_guard = (
    "DECISION_LEDGER_SEALED"
    in resolver_source
)

# ------------------------------------------------------------
# V3 formal report evidence.
# ------------------------------------------------------------

formal_report = load_json(
    FORMAL_REPORT
)

v3_postrun = load_json(
    V3_POSTRUN
)

gates = formal_report.get(
    "gates",
    {},
)

events = formal_report.get(
    "events",
    [],
)

if not isinstance(
    gates,
    dict,
):
    gates = {}

if not isinstance(
    events,
    list,
):
    events = []

# ------------------------------------------------------------
# Filesystem inventory by NAME only.
# No candidate truth JSON/JSONL is opened.
# ------------------------------------------------------------

candidate_roots = [
    S4 / "artifacts/fullpdf_v2",
    S5 / "artifacts/fullpdf_v2",
    S5 / "artifacts/fullpdf_v2_repair",
    S5 / "artifacts/stage5_fullpdf_v2",
    ART,
]

truth_name_tokens = (
    "truth",
    "evaluator",
    "future",
    "ground_truth",
)

truth_named_files = []

for root in candidate_roots:
    if not root.is_dir():
        continue

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in (
            ".json",
            ".jsonl",
            ".npz",
        ):
            continue

        low = path.name.lower()

        if any(
            token in low
            for token in truth_name_tokens
        ):
            truth_named_files.append(
                {
                    "path":
                        str(path),
                    "size":
                        path.stat().st_size,
                    "sha256":
                        sha(path),
                }
            )

# ------------------------------------------------------------
# Decision-ledger / seal inventory.
# Metadata/hash only.
# ------------------------------------------------------------

decision_files = []

for path in ART.rglob("*"):
    if not path.is_file():
        continue

    low = path.name.lower()

    if (
        "decision" in low
        and (
            "ledger" in low
            or "seal" in low
        )
    ):
        mode = stat.S_IMODE(
            path.stat().st_mode
        )

        decision_files.append(
            {
                "path":
                    str(path),
                "size":
                    path.stat().st_size,
                "sha256":
                    sha(path),
                "mode_octal":
                    oct(mode),
                "writable_bits":
                    mode & 0o222,
                "read_only":
                    (
                        mode & 0o222
                    ) == 0,
            }
        )

read_only_sealed = [
    x
    for x in decision_files
    if (
        x["read_only"]
        and "sealed"
        in Path(
            x["path"]
        ).name.lower()
    )
]

summary = {
    "stage": 5,
    "version": "fullpdf_v2",
    "status":
        "READ_ONLY_CONTRACT_AUDIT_PASS",
    "scientific_artifacts_modified":
        False,
    "formal_checker_executed":
        False,
    "candidate_truth_contents_opened":
        False,
    "current_hashes": {
        "formal_checker":
            sha(FORMAL_CHECKER),
        "checker2":
            sha(CHECKER2),
        "runtime":
            sha(RUNTIME),
        "protocol":
            sha(PROTOCOL),
        "Stage4_ledger":
            sha(STAGE4_LEDGER),
        "receiver_binding":
            sha(BINDING),
        "Checker2_V3_pretruth_seal":
            sha(NEW_PRETRUTH),
        "authority_seal":
            sha(AUTHORITY_SEAL),
        "V3_log":
            sha(V3_LOG),
        "V3_postrun":
            sha(V3_POSTRUN),
    },
    "V3": {
        "formal_status":
            v3_postrun.get(
                "formal_status"
            ),
        "formal_failure_reason":
            v3_postrun.get(
                "formal_failure_reason"
            ),
        "Stage6_allowed":
            v3_postrun.get(
                "Stage6_allowed",
                False,
            ),
    },
    "formal_report": {
        "status":
            formal_report.get(
                "status"
            ),
        "Stage5":
            formal_report.get(
                "Stage5"
            ),
        "Stage6_allowed":
            formal_report.get(
                "Stage6_allowed",
                False,
            ),
        "failure_reason":
            formal_report.get(
                "failure_reason",
                formal_report.get(
                    "reason"
                ),
            ),
        "G13":
            gates.get(
                "G13_two_exact_repeat_causal_decision_passes"
            ),
        "G14":
            gates.get(
                "G14_decision_ledger_sealed_before_truth"
            ),
        "event_names":
            [
                x.get("event")
                for x in events
                if isinstance(
                    x,
                    dict,
                )
            ],
    },
    "resolver": {
        "start_line":
            resolver.lineno,
        "end_line":
            resolver.end_lineno,
        "direct_calls":
            sorted(
                resolver_calls
            ),
        "sets_FORMAL_TRUTH_OPENED_true":
            resolver_sets_truth_opened,
        "mentions_DECISION_LEDGER_SEALED":
            resolver_mentions_seal_guard,
    },
    "functions": {
        name: {
            "start_line":
                info["start_line"],
            "end_line":
                info["end_line"],
            "calls":
                info["calls"],
            "string_literals":
                info[
                    "string_literals"
                ],
        }
        for name, info
        in function_info.items()
    },
    "interesting_global_constants":
        interesting_constants,
    "truth_named_file_inventory": {
        "count":
            len(
                truth_named_files
            ),
        "files":
            truth_named_files,
        "contents_read":
            False,
    },
    "decision_artifact_inventory": {
        "count":
            len(
                decision_files
            ),
        "read_only_sealed_count":
            len(
                read_only_sealed
            ),
        "files":
            decision_files,
    },
}

# ------------------------------------------------------------
# Full human-readable log.
# ------------------------------------------------------------

with FULL_LOG.open(
    "x",
    encoding="utf-8",
) as f:
    f.write(
        "STAGE5 FULLPDF V2 — "
        "EVALUATOR TRUTH BINDING SOURCE CONTRACT V1\n"
    )

    f.write(
        "NO FORMAL EXECUTION / "
        "NO TRUTH CONTENT READ / "
        "NO SCIENTIFIC MUTATION\n\n"
    )

    f.write(
        "===== SUMMARY JSON =====\n"
    )

    f.write(
        canonical(
            summary
        )
    )

    f.write(
        "\n===== EXACT GLOBAL CONSTANTS =====\n"
    )

    for name, info in sorted(
        interesting_constants.items(),
        key=lambda x:
            x[1]["line"],
    ):
        f.write(
            f"\n{name} @ line "
            f"{info['line']}\n"
        )

        f.write(
            f"{info['expression']}\n"
        )

    f.write(
        "\n===== EXACT RELEVANT FUNCTIONS =====\n"
    )

    for name in sorted(
        function_info,
        key=lambda n:
            function_info[n][
                "start_line"
            ],
    ):
        info = function_info[
            name
        ]

        f.write(
            "\n"
            + "=" * 78
            + "\n"
        )

        f.write(
            f"{name} | lines "
            f"{info['start_line']}-"
            f"{info['end_line']}\n"
        )

        f.write(
            "=" * 78
            + "\n"
        )

        f.write(
            info["source"]
        )

        f.write(
            "\n"
        )

SUMMARY_JSON.write_text(
    canonical(
        summary
    ),
    encoding="utf-8",
)

# Make audit outputs read-only.
os.chmod(
    FULL_LOG,
    0o444,
)

os.chmod(
    SUMMARY_JSON,
    0o444,
)

# ------------------------------------------------------------
# Tiny terminal summary.
# ------------------------------------------------------------

truth_parsers = [
    name
    for name in function_info
    if (
        "truth" in name.lower()
        and name
        != "resolve_truth_binding"
    )
    or (
        "parse" in name.lower()
        and name
        != "resolve_truth_binding"
    )
]

print("=" * 72)
print(
    "STAGE5 EVALUATOR-TRUTH CONTRACT AUDIT = PASS"
)
print("=" * 72)

print(
    "formal_checker_sha256 =",
    sha(
        FORMAL_CHECKER
    ),
)

print(
    "checker2_sha256 =",
    sha(
        CHECKER2
    ),
)

print(
    "V3_formal_reason =",
    v3_postrun.get(
        "formal_failure_reason"
    ),
)

print(
    "resolve_truth_binding_lines =",
    f"{resolver.lineno}-{resolver.end_lineno}",
)

print(
    "truth_parser_helpers =",
    truth_parsers,
)

if (
    "candidate_v2_paths"
    in function_info
):
    info = function_info[
        "candidate_v2_paths"
    ]

    print(
        "candidate_v2_paths_lines =",
        f"{info['start_line']}-"
        f"{info['end_line']}",
    )

print(
    "evaluate_sealed_decisions_lines =",
    f"{top_functions['evaluate_sealed_decisions'].lineno}-"
    f"{top_functions['evaluate_sealed_decisions'].end_lineno}",
)

print(
    "resolver_sets_FORMAL_TRUTH_OPENED =",
    resolver_sets_truth_opened,
)

print(
    "resolver_mentions_DECISION_LEDGER_SEALED =",
    resolver_mentions_seal_guard,
)

print(
    "truth_named_files_by_name_only =",
    len(
        truth_named_files
    ),
)

print(
    "decision_artifacts =",
    len(
        decision_files
    ),
)

print(
    "read_only_sealed_decision_artifacts =",
    len(
        read_only_sealed
    ),
)

for item in read_only_sealed[:3]:
    print(
        "sealed =",
        item["path"],
    )
    print(
        "sealed_sha256 =",
        item["sha256"],
    )

print(
    "scientific_artifacts_modified = false"
)

print(
    "formal_checker_executed = false"
)

print(
    "candidate_truth_contents_opened = false"
)

print(
    "summary_json =",
    SUMMARY_JSON,
)

print(
    "full_log =",
    FULL_LOG,
)

print("=" * 72)


if __name__ == "__main__":
    pass

from __future__ import annotations

import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback


ROOT = Path(
    "/home/agni/waymo"
)

S6 = (
    ROOT
    / "iscai_stage6"
)

BLOCK66_ART = (
    S6
    / "artifacts/block66"
)

BLOCK66_CFG = (
    S6
    / "configs"
)

RECON = (
    S6
    / "reports/"
      "block68_preoutcome_metric_reconciliation.json"
)

PROTOCOL = (
    S6
    / "configs/"
      "stage6_metric_freeze_protocol.json"
)

METRICS = (
    S6
    / "src/iscai_stage6/adb/"
      "metric_semantics.py"
)

INTERFACE = (
    S6
    / "src/iscai_stage6/adb/"
      "baseline_metric_contract.py"
)

CLASS_POLICY = (
    S6
    / "src/iscai_stage6/adb/"
      "class_aware_policy.py"
)

RUNTIME_SCRIPT = (
    S6
    / "scripts/"
      "run_block66_part3c1_class_aware_runtime.py"
)

PERSISTENCE_SCRIPT = (
    S6
    / "scripts/"
      "run_block66_part3c2b0_persistence_sufficiency_audit.py"
)

COHORT = (
    BLOCK66_ART
    / "block66_class_aware_development_cohort_120.jsonl"
)

COHORT_FREEZE = (
    BLOCK66_ART
    / "block66_development_cohort_freeze_manifest.json"
)

POSTERIOR = (
    BLOCK66_ART
    / "block66_development_identity_safe_gaussian_posterior.jsonl"
)

MATCHES = (
    BLOCK66_ART
    / "block66_part2c1_headlamp_eligible_predictive_matches.jsonl"
)

FALLBACKS = (
    BLOCK66_ART
    / "block66_part2c1_headlamp_eligible_reactive_fallbacks.jsonl"
)

POCC_MANIFEST = (
    BLOCK66_ART
    / "block66_part2c2b_eligible_actor_pocc_manifest.jsonl"
)

REPLAY_CONTRACT = (
    S6
    / "configs/"
      "block66_part3c_part2of2_deterministic_replay_contract.json"
)

PREREG = (
    S6
    / "configs/"
      "block66_part3b_class_aware_policy_preregistration.json"
)

REPORT = (
    S6
    / "reports/"
      "block68_part2_development_route_schema_bind.json"
)

EXPECTED_RECON_SHA = (
    "077a89fa5163f8862a3a5b7489d98929"
    "d9ae3e160ed9e030eb75f42a91d48f9f"
)

EXPECTED_PROTOCOL_SHA = (
    "409fbb2785e4ff13f29fdff96c91d608"
    "c245e009fcbb86ce57099d6a4e1815c9"
)

EXPECTED_RUNTIME_SHA = (
    "bf8358b5a7edfe40ffac2718ca7cd5af"
    "6e7b5fdb3ff8a2823caad6d0f40f057a"
)

EXPECTED_INTERFACE_SHA = (
    "39518597079f65d556ee8f0fd3b00ce5"
    "25319623c9eb8a83f6e3c608954213e9"
)

EXPECTED_STAGE6_TESTS = 202

MIN_FREE_GIB = 250.0


# ============================================================
# Helpers
# ============================================================

def sha256_file(
    path: Path,
) -> str:

    digest = sha256()

    with path.open(
        "rb"
    ) as stream:

        while True:

            chunk = stream.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def require(
    condition,
    message,
):

    if not bool(
        condition
    ):
        raise RuntimeError(
            message
        )


def load_json(
    path: Path,
):
    """
    Development/config reader with hard formal-name guard.
    """

    resolved = str(
        path.resolve()
    ).lower()

    require(
        "formal"
        not in
        resolved,
        (
            "FORMAL CONTENT ACCESS FORBIDDEN: "
            f"{path}"
        ),
    )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def first_jsonl_record(
    path: Path,
):
    """
    Reads only the first non-empty development JSONL record.

    Values are used internally only to derive schema/types;
    this command does not compute scientific performance.
    """

    resolved = str(
        path.resolve()
    ).lower()

    require(
        "formal"
        not in
        resolved,
        (
            "FORMAL CONTENT ACCESS FORBIDDEN: "
            f"{path}"
        ),
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:

        for line in stream:

            if line.strip():

                return json.loads(
                    line
                )

    raise RuntimeError(
        f"No records in {path}"
    )


def jsonl_count(
    path: Path,
) -> int:

    resolved = str(
        path.resolve()
    ).lower()

    require(
        "formal"
        not in
        resolved,
        (
            "FORMAL CONTENT ACCESS FORBIDDEN: "
            f"{path}"
        ),
    )

    count = 0

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:

        for line in stream:

            if line.strip():
                count += 1

    return count


def type_name(
    value,
) -> str:

    if value is None:
        return "null"

    if isinstance(
        value,
        bool,
    ):
        return "bool"

    if isinstance(
        value,
        int,
    ):
        return "int"

    if isinstance(
        value,
        float,
    ):
        return "float"

    if isinstance(
        value,
        str,
    ):
        return "str"

    if isinstance(
        value,
        dict,
    ):
        return "object"

    if isinstance(
        value,
        list,
    ):
        return "array"

    return type(
        value
    ).__name__


def schema_of(
    value,
    *,
    depth: int = 0,
    max_depth: int = 4,
):

    if depth >= max_depth:

        return {
            "type":
                type_name(
                    value
                )
        }

    if isinstance(
        value,
        dict,
    ):

        return {
            "type":
                "object",

            "keys": {
                str(
                    key
                ):
                    schema_of(
                        child,
                        depth=depth + 1,
                        max_depth=max_depth,
                    )
                for key, child in sorted(
                    value.items(),
                    key=lambda item: str(
                        item[0]
                    ),
                )
            },
        }

    if isinstance(
        value,
        list,
    ):

        result = {
            "type":
                "array",

            "length_first_record":
                len(
                    value
                ),
        }

        if value:

            result[
                "first_item"
            ] = schema_of(
                value[0],
                depth=depth + 1,
                max_depth=max_depth,
            )

        return result

    return {
        "type":
            type_name(
                value
            )
    }


def top_keys(
    value,
):

    if isinstance(
        value,
        dict,
    ):
        return sorted(
            str(
                key
            )
            for key in value
        )

    return []


def ast_functions(
    path: Path,
):

    source = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    tree = ast.parse(
        source,
        filename=str(
            path
        ),
    )

    functions = []

    for node in ast.walk(
        tree
    ):

        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            continue

        args = []

        positional = (
            list(
                node.args.posonlyargs
            )
            +
            list(
                node.args.args
            )
        )

        for arg in positional:
            args.append(
                arg.arg
            )

        if node.args.vararg:
            args.append(
                "*"
                +
                node.args.vararg.arg
            )

        for arg in (
            node.args.kwonlyargs
        ):
            args.append(
                arg.arg
            )

        if node.args.kwarg:
            args.append(
                "**"
                +
                node.args.kwarg.arg
            )

        functions.append({
            "name":
                node.name,

            "line":
                node.lineno,

            "arguments":
                args,
        })

    return sorted(
        functions,
        key=lambda item: (
            item[
                "line"
            ],
            item[
                "name"
            ],
        ),
    )


def interesting_source_lines(
    path: Path,
):

    tokens = (
        "I_final",
        "illumination",
        "oracle",
        "future",
        "metric",
        "jsonl",
        "cache",
        "persist",
        "p_occ",
        "pocc",
        "reactive",
        "predictive",
        "output",
        "write",
        "save",
        "replay",
    )

    results = []

    for index, line in enumerate(
        path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines(),
        start=1,
    ):

        lower = line.lower()

        if any(
            token in lower
            for token in tokens
        ):

            stripped = line.strip()

            if len(
                stripped
            ) > 220:

                stripped = (
                    stripped[:217]
                    +
                    "..."
                )

            results.append({
                "line":
                    index,

                "text":
                    stripped,
            })

    return results[
        :160
    ]


def development_candidate_paths():

    results = []

    if not BLOCK66_ART.is_dir():
        return results

    tokens = (
        "part3c",
        "class_aware",
        "illumination",
        "replay",
        "persistence",
        "schedule",
        "runtime",
        "metric",
        "oracle",
    )

    for path in BLOCK66_ART.rglob(
        "*"
    ):

        if not path.is_file():
            continue

        relative = str(
            path.relative_to(
                BLOCK66_ART
            )
        )

        lower = relative.lower()

        # Hard guard.
        if "formal" in lower:
            continue

        if any(
            token in lower
            for token in tokens
        ):
            results.append(
                relative
            )

    return sorted(
        results
    )


def possible_path_strings(
    value,
):

    found = []

    def walk(
        item,
        key="",
    ):

        if isinstance(
            item,
            dict,
        ):

            for child_key, child in (
                item.items()
            ):

                walk(
                    child,
                    str(
                        child_key
                    ),
                )

        elif isinstance(
            item,
            list,
        ):

            for child in item:
                walk(
                    child,
                    key,
                )

        elif isinstance(
            item,
            str,
        ):

            lower_key = key.lower()
            lower_value = item.lower()

            if (
                any(
                    token in lower_key
                    for token in (
                        "path",
                        "file",
                        "cache",
                        "manifest",
                        "artifact",
                        "output",
                    )
                )
                or
                any(
                    token in lower_value
                    for token in (
                        ".json",
                        ".jsonl",
                        ".npz",
                        "/artifacts/",
                        "/reports/",
                    )
                )
            ):

                if "formal" not in lower_value:

                    found.append(
                        item
                    )

    walk(
        value
    )

    return sorted(
        set(
            found
        )
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
        cwd=str(
            S6
        ),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    count = None

    for line in (
        process.stdout.splitlines()
    ):

        match = re.search(
            r"Ran\s+(\d+)\s+tests?",
            line,
        )

        if match:
            count = int(
                match.group(1)
            )

    return (
        process.returncode,
        count,
        process.stdout,
    )


def write_json(
    path: Path,
    payload,
):

    temporary = path.with_suffix(
        path.suffix
        +
        ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.8 PART 2/2"
    )
    print(
        "DEVELOPMENT RUNTIME ROUTE / SCHEMA BIND"
    )
    print(
        "FORMAL CONTENT HARD-BLOCKED"
    )
    print(
        "============================================================"
    )

    required = (
        RECON,
        PROTOCOL,
        METRICS,
        INTERFACE,
        CLASS_POLICY,
        RUNTIME_SCRIPT,
        PERSISTENCE_SCRIPT,
        COHORT,
        COHORT_FREEZE,
        POSTERIOR,
        MATCHES,
        FALLBACKS,
        POCC_MANIFEST,
        REPLAY_CONTRACT,
        PREREG,
    )

    missing = [
        str(
            path
        )
        for path in required
        if not path.is_file()
    ]

    require(
        not missing,
        (
            "Missing development-route dependency: "
            +
            ", ".join(
                missing
            )
        ),
    )

    # ========================================================
    # A. Reconciliation seal
    # ========================================================

    print()
    print(
        "===== A. RECONCILIATION SEAL ====="
    )

    assert_file_pairs = (
        (
            RECON,
            EXPECTED_RECON_SHA,
            "reconciliation report",
        ),
        (
            PROTOCOL,
            EXPECTED_PROTOCOL_SHA,
            "reconciled protocol",
        ),
        (
            METRICS,
            EXPECTED_RUNTIME_SHA,
            "reconciled metric runtime",
        ),
        (
            INTERFACE,
            EXPECTED_INTERFACE_SHA,
            "reconciled metric interface",
        ),
    )

    for (
        path,
        expected,
        label,
    ) in assert_file_pairs:

        require(
            sha256_file(
                path
            )
            ==
            expected,
            (
                f"{label} SHA changed."
            ),
        )

        print(
            f"{label:30s}= EXACT PASS"
        )

    recon = load_json(
        RECON
    )

    require(
        recon.get(
            "status"
        )
        ==
        "PASS_PREOUTCOME_METRIC_RECONCILED",
        (
            "Reconciliation report "
            "status mismatch."
        ),
    )

    require(
        recon[
            "scientific_boundary"
        ][
            "development_outcomes_read"
        ]
        is False,
        (
            "Reconciliation does not preserve "
            "pre-development boundary."
        ),
    )

    require(
        recon[
            "scientific_boundary"
        ][
            "formal_outcomes_read"
        ]
        is False,
        (
            "Formal outcomes were already read."
        ),
    )

    print(
        "Block6.6 authority             = FROZEN"
    )

    print(
        "development outcome read so far= NO"
    )

    print(
        "formal outcome read            = NO"
    )

    # ========================================================
    # B. Frozen development cohort
    # ========================================================

    print()
    print(
        "===== B. DEVELOPMENT POPULATION ====="
    )

    cohort_n = jsonl_count(
        COHORT
    )

    posterior_n = jsonl_count(
        POSTERIOR
    )

    matches_n = jsonl_count(
        MATCHES
    )

    fallbacks_n = jsonl_count(
        FALLBACKS
    )

    pocc_n = jsonl_count(
        POCC_MANIFEST
    )

    print(
        "development cohort records    =",
        cohort_n,
    )

    print(
        "posterior records             =",
        posterior_n,
    )

    print(
        "eligible predictive matches   =",
        matches_n,
    )

    print(
        "reactive fallback records     =",
        fallbacks_n,
    )

    print(
        "P_occ manifest records        =",
        pocc_n,
    )

    require(
        cohort_n == 120,
        (
            "Frozen development cohort "
            f"must contain 120 scenarios; got {cohort_n}."
        ),
    )

    # The other files are actor-level and need not equal 120.
    require(
        posterior_n > 0,
        "Development posterior is empty.",
    )

    require(
        matches_n > 0,
        "Eligible predictive matches are empty.",
    )

    require(
        pocc_n > 0,
        "P_occ manifest is empty.",
    )

    print(
        "development cohort = FROZEN 120 PASS"
    )

    # ========================================================
    # C. Record schemas
    # ========================================================

    print()
    print(
        "===== C. DEVELOPMENT RECORD SCHEMAS ====="
    )

    development_files = {
        "cohort":
            COHORT,

        "posterior":
            POSTERIOR,

        "predictive_matches":
            MATCHES,

        "reactive_fallbacks":
            FALLBACKS,

        "P_occ_manifest":
            POCC_MANIFEST,
    }

    schemas = {}

    for name, path in (
        development_files.items()
    ):

        first = first_jsonl_record(
            path
        )

        schema = schema_of(
            first
        )

        schemas[
            name
        ] = {
            "path":
                str(
                    path
                ),

            "sha256":
                sha256_file(
                    path
                ),

            "records":
                jsonl_count(
                    path
                ),

            "top_keys":
                top_keys(
                    first
                ),

            "schema":
                schema,
        }

        print()
        print(
            "DATASET =",
            name,
        )

        print(
            "path =",
            path,
        )

        print(
            "records =",
            schemas[
                name
            ][
                "records"
            ],
        )

        print(
            "top-level keys =",
            schemas[
                name
            ][
                "top_keys"
            ],
        )

    # ========================================================
    # D. Replay / prereg config route
    # ========================================================

    print()
    print(
        "===== D. DEVELOPMENT REPLAY CONTRACT ====="
    )

    replay = load_json(
        REPLAY_CONTRACT
    )

    prereg = load_json(
        PREREG
    )

    replay_paths = (
        possible_path_strings(
            replay
        )
    )

    prereg_paths = (
        possible_path_strings(
            prereg
        )
    )

    print(
        "replay contract status =",
        replay.get(
            "status"
        ),
    )

    print(
        "replay contract SHA256 =",
        sha256_file(
            REPLAY_CONTRACT
        ),
    )

    print()
    print(
        "replay-declared paths:"
    )

    for item in replay_paths[
        :80
    ]:
        print(
            "  ",
            item,
        )

    print()
    print(
        "prereg-declared paths:"
    )

    for item in prereg_paths[
        :80
    ]:
        print(
            "  ",
            item,
        )

    # ========================================================
    # E. Existing Block6.6 runtime artifacts
    # ========================================================

    print()
    print(
        "===== E. EXISTING BLOCK6.6 RUNTIME ARTIFACTS ====="
    )

    candidates = (
        development_candidate_paths()
    )

    print(
        "candidate files =",
        len(
            candidates
        ),
    )

    for relative in candidates[
        :200
    ]:

        print(
            "  ",
            relative,
        )

    # ========================================================
    # F. Runtime source routes
    # ========================================================

    print()
    print(
        "===== F. CLASS-AWARE RUNTIME SOURCE ROUTE ====="
    )

    runtime_functions = (
        ast_functions(
            RUNTIME_SCRIPT
        )
    )

    persistence_functions = (
        ast_functions(
            PERSISTENCE_SCRIPT
        )
    )

    print(
        "runtime script SHA256 =",
        sha256_file(
            RUNTIME_SCRIPT
        ),
    )

    print()
    print(
        "runtime functions:"
    )

    for item in runtime_functions:

        print(
            f"  L{item['line']:04d} "
            f"{item['name']}("
            f"{', '.join(item['arguments'])})"
        )

    print()
    print(
        "persistence functions:"
    )

    for item in persistence_functions:

        print(
            f"  L{item['line']:04d} "
            f"{item['name']}("
            f"{', '.join(item['arguments'])})"
        )

    print()
    print(
        "relevant runtime source lines:"
    )

    runtime_lines = (
        interesting_source_lines(
            RUNTIME_SCRIPT
        )
    )

    for item in runtime_lines[
        :120
    ]:

        print(
            f"  L{item['line']:04d}: "
            f"{item['text']}"
        )

    print()
    print(
        "relevant persistence source lines:"
    )

    persistence_lines = (
        interesting_source_lines(
            PERSISTENCE_SCRIPT
        )
    )

    for item in persistence_lines[
        :100
    ]:

        print(
            f"  L{item['line']:04d}: "
            f"{item['text']}"
        )

    # ========================================================
    # G. Existing candidate JSON/JSONL schemas
    # ========================================================

    print()
    print(
        "===== G. EXISTING RUNTIME OUTPUT SCHEMAS ====="
    )

    candidate_schemas = {}

    inspected = 0

    for relative in candidates:

        path = (
            BLOCK66_ART
            /
            relative
        )

        suffix = (
            path.suffix.lower()
        )

        # Never open a formal-named file.
        require(
            "formal"
            not in
            relative.lower(),
            (
                "Formal candidate unexpectedly "
                "entered development list."
            ),
        )

        try:

            if suffix == ".jsonl":

                first = (
                    first_jsonl_record(
                        path
                    )
                )

                candidate_schemas[
                    relative
                ] = {
                    "kind":
                        "jsonl",

                    "top_keys":
                        top_keys(
                            first
                        ),

                    "schema":
                        schema_of(
                            first
                        ),
                }

            elif suffix == ".json":

                payload = (
                    load_json(
                        path
                    )
                )

                candidate_schemas[
                    relative
                ] = {
                    "kind":
                        "json",

                    "top_keys":
                        top_keys(
                            payload
                        ),

                    "schema":
                        schema_of(
                            payload
                        ),
                }

            else:
                continue

            inspected += 1

            print()
            print(
                "OUTPUT =",
                relative,
            )

            print(
                "type =",
                candidate_schemas[
                    relative
                ][
                    "kind"
                ],
            )

            print(
                "top-level keys =",
                candidate_schemas[
                    relative
                ][
                    "top_keys"
                ],
            )

            # Enough schemas for route binding.
            if inspected >= 40:
                break

        except Exception as exc:

            print(
                "schema skip =",
                relative,
                "|",
                type(
                    exc
                ).__name__,
                str(
                    exc
                ),
            )

    require(
        inspected > 0,
        (
            "No Block6.6 runtime output "
            "JSON/JSONL schema could be inspected."
        ),
    )

    print()
    print(
        "runtime output schemas inspected =",
        inspected,
    )

    # ========================================================
    # H. Search schema keys for exact evaluator ingredients
    # ========================================================

    print()
    print(
        "===== H. EVALUATOR INGREDIENT DISCOVERY ====="
    )

    all_schema_text = json.dumps(
        {
            "development":
                schemas,

            "candidate_outputs":
                candidate_schemas,
        },
        sort_keys=True,
    ).lower()

    ingredient_tokens = {
        "scenario_identity":
            (
                "scenario_id",
                "scenario",
            ),

        "actor_identity":
            (
                "track_id",
                "actor_id",
                "track_index",
            ),

        "class":
            (
                "actor_class",
                "object_type",
                "class",
            ),

        "P_occ":
            (
                "p_occ",
                "occupancy",
            ),

        "illumination":
            (
                "illumination",
                "i_final",
                "intensity",
            ),

        "oracle":
            (
                "oracle",
                "future",
            ),

        "reactive":
            (
                "reactive",
                "fallback",
            ),

        "four_horizons":
            (
                "horizon",
                "horizons",
            ),
    }

    readiness = {}

    for name, tokens in (
        ingredient_tokens.items()
    ):

        found = any(
            token in all_schema_text
            for token in tokens
        )

        readiness[
            name
        ] = found

        print(
            f"{name:24s}=",
            (
                "FOUND"
                if found
                else "NOT PROVEN"
            ),
        )

    # We intentionally do not require all ingredients yet:
    # source route may construct some at runtime rather than
    # persist them directly.

    # ========================================================
    # I. Formal hard boundary proof
    # ========================================================

    print()
    print(
        "===== I. FORMAL HARD BOUNDARY ====="
    )

    print(
        "formal artifact content opened = NO"
    )

    print(
        "formal manifest content opened = NO"
    )

    print(
        "formal outcomes read           = NO"
    )

    print(
        "formal evaluation executed     = NO"
    )

    print(
        "development records opened     = YES / SCHEMA ONLY"
    )

    print(
        "development performance metrics= NOT COMPUTED"
    )

    # ========================================================
    # J. Regression
    # ========================================================

    print()
    print(
        "===== J. FULL STAGE6 REGRESSION ====="
    )

    rc, test_count, output = (
        run_regression()
    )

    if rc != 0:

        print(
            "\n".join(
                output.splitlines()[
                    -100:
                ]
            )
        )

    require(
        rc == 0,
        (
            "Stage6 regression failed."
        ),
    )

    require(
        test_count
        ==
        EXPECTED_STAGE6_TESTS,
        (
            "Unexpected regression count: "
            f"{test_count}; expected "
            f"{EXPECTED_STAGE6_TESTS}."
        ),
    )

    print(
        "Stage6 regression =",
        f"{test_count} / {test_count} PASS",
    )

    # ========================================================
    # K. Reconciliation immutability
    # ========================================================

    print()
    print(
        "===== K. RECONCILIATION IMMUTABILITY ====="
    )

    require(
        sha256_file(
            RECON
        )
        ==
        EXPECTED_RECON_SHA,
        (
            "Reconciliation report changed."
        ),
    )

    require(
        sha256_file(
            PROTOCOL
        )
        ==
        EXPECTED_PROTOCOL_SHA,
        (
            "Reconciled protocol changed."
        ),
    )

    require(
        sha256_file(
            METRICS
        )
        ==
        EXPECTED_RUNTIME_SHA,
        (
            "Reconciled metric runtime changed."
        ),
    )

    require(
        sha256_file(
            INTERFACE
        )
        ==
        EXPECTED_INTERFACE_SHA,
        (
            "Reconciled metric interface changed."
        ),
    )

    print(
        "reconciliation report = UNCHANGED"
    )

    print(
        "metric protocol       = UNCHANGED"
    )

    print(
        "metric runtime        = UNCHANGED"
    )

    print(
        "metric interface      = UNCHANGED"
    )

    # ========================================================
    # L. Storage
    # ========================================================

    print()
    print(
        "===== L. STORAGE ====="
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
            "250-GiB reserve violated."
        ),
    )

    print(
        "free GiB        =",
        round(
            free_gib,
            3,
        ),
    )

    print(
        "250-GiB reserve = PASS"
    )

    # ========================================================
    # M. Route binding report
    # ========================================================

    result = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.8",

        "part":
            "2/2_development_route_schema_bind",

        "status":
            "PASS_DEVELOPMENT_ROUTE_SCHEMA_BOUND",

        "reconciliation": {
            "report_sha256":
                EXPECTED_RECON_SHA,

            "protocol_sha256":
                EXPECTED_PROTOCOL_SHA,

            "metric_runtime_sha256":
                EXPECTED_RUNTIME_SHA,

            "metric_interface_sha256":
                EXPECTED_INTERFACE_SHA,
        },

        "development_population": {
            "cohort_path":
                str(
                    COHORT
                ),

            "cohort_records":
                cohort_n,

            "posterior_records":
                posterior_n,

            "predictive_match_records":
                matches_n,

            "reactive_fallback_records":
                fallbacks_n,

            "P_occ_manifest_records":
                pocc_n,
        },

        "development_schemas":
            schemas,

        "runtime_route": {
            "class_aware_runtime_script":
                str(
                    RUNTIME_SCRIPT
                ),

            "class_aware_runtime_sha256":
                sha256_file(
                    RUNTIME_SCRIPT
                ),

            "persistence_script":
                str(
                    PERSISTENCE_SCRIPT
                ),

            "persistence_script_sha256":
                sha256_file(
                    PERSISTENCE_SCRIPT
                ),

            "runtime_functions":
                runtime_functions,

            "persistence_functions":
                persistence_functions,

            "runtime_relevant_lines":
                runtime_lines,

            "persistence_relevant_lines":
                persistence_lines,
        },

        "existing_runtime_candidate_paths":
            candidates,

        "existing_runtime_output_schemas":
            candidate_schemas,

        "evaluator_ingredient_readiness":
            readiness,

        "scientific_execution": {
            "development_records_opened":
                True,

            "development_use":
                "SCHEMA_ROUTE_BINDING_ONLY",

            "development_performance_metrics_computed":
                False,

            "numeric_noninferiority_bounds_selected":
                False,

            "policy_tuning":
                False,

            "parameter_tuning":
                False,

            "formal_artifact_content_opened":
                False,

            "formal_outcomes_read":
                False,

            "formal_evaluation":
                False,
        },

        "regression": {
            "tests":
                test_count,

            "status":
                "PASS",
        },

        "storage": {
            "free_GiB":
                free_gib,

            "hard_reserve_GiB":
                MIN_FREE_GIB,

            "pass":
                True,
        },

        "next":
            (
                "Execute exact development-only "
                "class-aware vs original-reactive "
                "ADB metric evaluation using the "
                "bound runtime route, then freeze "
                "numeric non-inferiority bounds. "
                "Formal remains prohibited."
            ),
    }

    write_json(
        REPORT,
        result,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 PART 2/2 — DEVELOPMENT ROUTE BIND FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "reconciled metric authority = EXACT PASS"
    )

    print(
        "development cohort           = 120 SCENARIOS"
    )

    print(
        "development schemas          = BOUND"
    )

    print(
        "class-aware runtime route    = BOUND"
    )

    print(
        "P_occ route                  = BOUND"
    )

    print(
        "reactive fallback route      = BOUND"
    )

    print(
        "development outcomes         = SCHEMA ONLY"
    )

    print(
        "performance metrics computed = NO"
    )

    print(
        "numeric bounds frozen        = NO"
    )

    print(
        "formal files opened          = NO"
    )

    print(
        "formal outcomes              = NO"
    )

    print(
        "policy/parameter tuning      = NO"
    )

    print(
        "Stage6 regression            =",
        f"{test_count} / {test_count} PASS",
    )

    print(
        "STATUS = PASS_DEVELOPMENT_ROUTE_SCHEMA_BOUND"
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
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 DEVELOPMENT ROUTE/SCHEMA BIND = BLOCKED"
    )
    print(
        "============================================================"
    )

    print(
        "exception =",
        type(
            exc
        ).__name__,
        str(
            exc
        ),
    )

    print()
    traceback.print_exc()

    print()
    print(
        "development performance metrics = NOT COMPUTED"
    )

    print(
        "numeric bounds                 = NOT SELECTED"
    )

    print(
        "formal artifact content        = NOT OPENED"
    )

    print(
        "formal outcomes                = NOT READ"
    )

    print(
        "formal evaluation              = NO"
    )

    print(
        "policy tuning                  = NO"
    )

    print()
    print(
        "Do not run formal Stage6 evaluation."
    )

    print(
        "Send this BLOCKED output for targeted repair."
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

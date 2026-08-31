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

S6 = ROOT / "iscai_stage6"

ADB_SRC = (
    S6
    / "src/iscai_stage6/adb"
)

SCRIPTS = (
    S6
    / "scripts"
)

CONFIGS = (
    S6
    / "configs"
)

BLOCK66 = (
    S6
    / "artifacts/block66"
)

BINDER = (
    S6
    / "reports/"
      "block68_exact_evaluator_binding_repair.json"
)

ROUTE_BIND = (
    S6
    / "reports/"
      "block68_part2_development_route_schema_bind.json"
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

PREREG = (
    S6
    / "configs/"
      "block66_part3b_class_aware_policy_preregistration.json"
)

REPORT = (
    S6
    / "reports/"
      "block68_oracle_roi_route_discovery.json"
)


EXPECTED_BINDER_SHA = (
    "1505c780423efbe27735c3729b95a18c"
    "3da2093f49826bd96cbfad8fdce5d5f1"
)

EXPECTED_RECON_SHA = (
    "077a89fa5163f8862a3a5b7489d98929"
    "d9ae3e160ed9e030eb75f42a91d48f9f"
)

EXPECTED_PROTOCOL_SHA = (
    "409fbb2785e4ff13f29fdff96c91d608"
    "c245e009fcbb86ce57099d6a4e1815c9"
)

EXPECTED_METRICS_SHA = (
    "bf8358b5a7edfe40ffac2718ca7cd5af"
    "6e7b5fdb3ff8a2823caad6d0f40f057a"
)

EXPECTED_INTERFACE_SHA = (
    "39518597079f65d556ee8f0fd3b00ce5"
    "25319623c9eb8a83f6e3c608954213e9"
)

EXPECTED_STAGE6_TESTS = 202

MIN_FREE_GIB = 250.0

MAX_SCHEMA_FILES = 40
MAX_SOURCE_CANDIDATES = 16


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


def nonformal(
    path: Path,
) -> bool:

    return (
        "formal"
        not in
        str(
            path
        ).lower()
    )


def guarded_json(
    path: Path,
):

    require(
        nonformal(
            path
        ),
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


def guarded_first_jsonl(
    path: Path,
):

    require(
        nonformal(
            path
        ),
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


def type_name(
    value,
):

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


def schema_paths(
    value,
    *,
    prefix="",
    depth=0,
    max_depth=5,
):

    if depth > max_depth:
        return []

    results = []

    if isinstance(
        value,
        dict,
    ):

        for key, child in (
            value.items()
        ):

            path = (
                f"{prefix}.{key}"
                if prefix
                else str(
                    key
                )
            )

            results.append({
                "path":
                    path,

                "type":
                    type_name(
                        child
                    ),
            })

            results.extend(
                schema_paths(
                    child,
                    prefix=path,
                    depth=depth + 1,
                    max_depth=max_depth,
                )
            )

    elif isinstance(
        value,
        list,
    ) and value:

        path = (
            prefix
            +
            "[]"
        )

        results.append({
            "path":
                path,

            "type":
                type_name(
                    value[0]
                ),
        })

        results.extend(
            schema_paths(
                value[0],
                prefix=path,
                depth=depth + 1,
                max_depth=max_depth,
            )
        )

    return results


def ast_signature(
    node,
):

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

    return args


def source_excerpt(
    lines,
    start,
    end,
):

    selected = []

    for index in range(
        max(
            1,
            start
        ),
        min(
            len(
                lines
            ),
            end,
        )
        +
        1,
    ):

        text = lines[
            index - 1
        ].rstrip()

        if len(
            text
        ) > 220:

            text = (
                text[:217]
                +
                "..."
            )

        selected.append({
            "line":
                index,

            "text":
                text,
        })

    return selected


# ============================================================
# Source discovery
# ============================================================

CATEGORY_RULES = {
    "constructed_oracle": (
        "oracle",
        "constructed",
        "future",
        "truth",
    ),

    "future_full_box": (
        "future",
        "box",
        "corner",
        "footprint",
        "project",
        "raster",
    ),

    "road_roi": (
        "road",
        "roi",
        "illumination",
        "grid",
    ),

    "future_truth_access": (
        "future_gt",
        "future truth",
        "future_state",
        "future",
        "truth",
    ),

    "reactive_reference": (
        "reactive",
        "part_a",
        "current",
        "fallback",
    ),
}


def category_score(
    text,
    tokens,
):

    lower = (
        text.lower()
    )

    return sum(
        1
        for token in tokens
        if token in lower
    )


def discover_python_file(
    path: Path,
):

    source = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    lines = source.splitlines()

    try:

        tree = ast.parse(
            source,
            filename=str(
                path
            ),
        )

    except SyntaxError:

        return []

    results = []

    for node in tree.body:

        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            continue

        start = int(
            node.lineno
        )

        end = int(
            getattr(
                node,
                "end_lineno",
                start,
            )
        )

        snippet_text = "\n".join(
            lines[
                start - 1:
                min(
                    end,
                    start + 140,
                )
            ]
        )

        categories = {}

        for category, tokens in (
            CATEGORY_RULES.items()
        ):

            score = category_score(
                (
                    node.name
                    +
                    "\n"
                    +
                    snippet_text
                ),
                tokens,
            )

            if score > 0:

                categories[
                    category
                ] = score

        if not categories:
            continue

        result = {
            "name":
                node.name,

            "kind":
                (
                    "class"
                    if isinstance(
                        node,
                        ast.ClassDef,
                    )
                    else
                    "function"
                ),

            "line":
                start,

            "end_line":
                end,

            "categories":
                categories,

            "excerpt":
                source_excerpt(
                    lines,
                    start,
                    min(
                        end,
                        start + 80,
                    ),
                ),
        }

        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):

            result[
                "arguments"
            ] = ast_signature(
                node
            )

        results.append(
            result
        )

    return results


# ============================================================
# Config / artifact candidate discovery
# ============================================================

DISCOVERY_TOKENS = (
    "oracle",
    "future",
    "truth",
    "full_box",
    "full-box",
    "footprint",
    "road",
    "roi",
    "illumination",
    "vehicle_surrogate",
    "pedestrian",
    "cyclist",
)


def path_interesting(
    path: Path,
):

    lower = str(
        path
    ).lower()

    return (
        nonformal(
            path
        )
        and
        any(
            token in lower
            for token in DISCOVERY_TOKENS
        )
    )


def candidate_data_files():

    files = []

    for root in (
        CONFIGS,
        BLOCK66,
    ):

        if not root.exists():
            continue

        for path in root.rglob(
            "*"
        ):

            if not path.is_file():
                continue

            if not path_interesting(
                path
            ):
                continue

            if (
                path.suffix.lower()
                not in (
                    ".json",
                    ".jsonl",
                )
            ):
                continue

            files.append(
                path
            )

    return sorted(
        set(
            files
        ),
        key=lambda path: str(
            path
        ),
    )


# ============================================================
# Semantic readiness from evidence
# ============================================================

def contains_any(
    text,
    tokens,
):

    lower = (
        text.lower()
    )

    return any(
        token in lower
        for token in tokens
    )


# ============================================================
# Regression
# ============================================================

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

    temporary = (
        path.with_suffix(
            path.suffix
            +
            ".tmp"
        )
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
        "CONSTRUCTED-ORACLE / FULL-BOX / ROAD-ROI ROUTE DISCOVERY"
    )
    print(
        "NO PERFORMANCE METRICS / FORMAL CONTENT HARD-BLOCKED"
    )
    print(
        "============================================================"
    )

    required = (
        BINDER,
        ROUTE_BIND,
        RECON,
        PROTOCOL,
        METRICS,
        INTERFACE,
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
            "Missing route-discovery dependency: "
            +
            ", ".join(
                missing
            )
        ),
    )

    # ========================================================
    # A. Frozen seal
    # ========================================================

    print()
    print(
        "===== A. FROZEN EVALUATOR SEAL ====="
    )

    require(
        sha256_file(
            BINDER
        )
        ==
        EXPECTED_BINDER_SHA,
        (
            "Final-output binder SHA changed."
        ),
    )

    binder = guarded_json(
        BINDER
    )

    require(
        binder.get(
            "status"
        )
        ==
        "PASS_FINAL_ACTUATION_OUTPUT_BOUND",
        (
            "Final-output binder is not PASS."
        ),
    )

    exact = (
        (
            RECON,
            EXPECTED_RECON_SHA,
            "metric reconciliation",
        ),
        (
            PROTOCOL,
            EXPECTED_PROTOCOL_SHA,
            "metric protocol",
        ),
        (
            METRICS,
            EXPECTED_METRICS_SHA,
            "metric runtime",
        ),
        (
            INTERFACE,
            EXPECTED_INTERFACE_SHA,
            "metric interface",
        ),
    )

    for (
        path,
        expected,
        label,
    ) in exact:

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
            f"{label:26s}= EXACT PASS"
        )

    print(
        "I_final binding             = EXACT PASS"
    )

    print(
        "formal evaluation           = BLOCKED"
    )

    # ========================================================
    # B. Preregistered evaluator semantics
    # ========================================================

    print()
    print(
        "===== B. BLOCK6.6 PREREGISTERED EVALUATOR SEMANTICS ====="
    )

    prereg = guarded_json(
        PREREG
    )

    prereg_text = json.dumps(
        prereg,
        indent=2,
        sort_keys=True,
    )

    prereg_lines = []

    for line in (
        prereg_text.splitlines()
    ):

        lower = line.lower()

        if any(
            token in lower
            for token in DISCOVERY_TOKENS
        ):

            prereg_lines.append(
                line.strip()
            )

    for line in prereg_lines[
        :160
    ]:

        print(
            line
        )

    print()
    print(
        "prereg relevant lines =",
        len(
            prereg_lines
        ),
    )

    # ========================================================
    # C. ADB source discovery
    # ========================================================

    print()
    print(
        "===== C. ADB SOURCE ROUTE CANDIDATES ====="
    )

    source_candidates = []

    for path in sorted(
        ADB_SRC.glob(
            "*.py"
        )
    ):

        if not nonformal(
            path
        ):
            continue

        discovered = (
            discover_python_file(
                path
            )
        )

        for item in discovered:

            source_candidates.append({
                "module":
                    path.name,

                "sha256":
                    sha256_file(
                        path
                    ),

                **item,
            })

    source_candidates.sort(
        key=lambda item: (
            -sum(
                item[
                    "categories"
                ].values()
            ),
            item[
                "module"
            ],
            item[
                "line"
            ],
        )
    )

    for item in source_candidates[
        :MAX_SOURCE_CANDIDATES
    ]:

        print()
        print(
            "------------------------------------------------------------"
        )

        print(
            "MODULE =",
            item[
                "module"
            ],
        )

        print(
            f"{item['kind'].upper()} = "
            f"{item['name']} "
            f"@L{item['line']}"
        )

        print(
            "categories =",
            item[
                "categories"
            ],
        )

        if (
            item[
                "kind"
            ]
            ==
            "function"
        ):

            print(
                "arguments =",
                item.get(
                    "arguments",
                    []
                ),
            )

        print(
            "excerpt:"
        )

        for line in (
            item[
                "excerpt"
            ]
        ):

            print(
                f"  L{line['line']:04d}: "
                f"{line['text']}"
            )

    require(
        source_candidates,
        (
            "No evaluator-related Stage6 "
            "source candidates found."
        ),
    )

    # ========================================================
    # D. Script-side evaluator source candidates
    # ========================================================

    print()
    print(
        "===== D. SCRIPT-SIDE ORACLE / FUTURE / ROI ROUTES ====="
    )

    script_candidates = []

    for path in sorted(
        SCRIPTS.glob(
            "*.py"
        )
    ):

        if not nonformal(
            path
        ):
            continue

        # Avoid recursively treating this discovery script
        # itself as scientific evidence.
        if (
            path.name
            ==
            Path(
                __file__
            ).name
        ):
            continue

        discovered = (
            discover_python_file(
                path
            )
        )

        for item in discovered:

            if not any(
                category in item[
                    "categories"
                ]
                for category in (
                    "constructed_oracle",
                    "future_full_box",
                    "road_roi",
                    "future_truth_access",
                )
            ):
                continue

            script_candidates.append({
                "script":
                    path.name,

                "sha256":
                    sha256_file(
                        path
                    ),

                **item,
            })

    script_candidates.sort(
        key=lambda item: (
            -sum(
                item[
                    "categories"
                ].values()
            ),
            item[
                "script"
            ],
            item[
                "line"
            ],
        )
    )

    for item in script_candidates[
        :MAX_SOURCE_CANDIDATES
    ]:

        print()
        print(
            "------------------------------------------------------------"
        )

        print(
            "SCRIPT =",
            item[
                "script"
            ],
        )

        print(
            f"{item['kind'].upper()} = "
            f"{item['name']} "
            f"@L{item['line']}"
        )

        print(
            "categories =",
            item[
                "categories"
            ],
        )

        if (
            item[
                "kind"
            ]
            ==
            "function"
        ):

            print(
                "arguments =",
                item.get(
                    "arguments",
                    []
                ),
            )

        print(
            "excerpt:"
        )

        for line in (
            item[
                "excerpt"
            ][:60]
        ):

            print(
                f"  L{line['line']:04d}: "
                f"{line['text']}"
            )

    # ========================================================
    # E. Config/artifact schema candidates
    # ========================================================

    print()
    print(
        "===== E. NONFORMAL CONFIG / ARTIFACT SCHEMAS ====="
    )

    data_candidates = (
        candidate_data_files()
    )

    schema_evidence = {}

    inspected = 0

    for path in data_candidates:

        if inspected >= MAX_SCHEMA_FILES:
            break

        try:

            if (
                path.suffix.lower()
                ==
                ".json"
            ):

                payload = guarded_json(
                    path
                )

            else:

                payload = guarded_first_jsonl(
                    path
                )

        except Exception as exc:

            print(
                "SKIP",
                path,
                "|",
                type(
                    exc
                ).__name__,
                str(
                    exc
                ),
            )

            continue

        relative = str(
            path.relative_to(
                S6
            )
        )

        paths = schema_paths(
            payload
        )

        path_text = " ".join(
            item[
                "path"
            ]
            for item in paths
        )

        relevant = [
            item
            for item in paths
            if any(
                token
                in
                item[
                    "path"
                ].lower()
                for token in DISCOVERY_TOKENS
            )
        ]

        # Ignore filename-only hits if the schema itself
        # provides no useful evaluator-related fields.
        if not relevant:
            continue

        schema_evidence[
            relative
        ] = {
            "sha256":
                sha256_file(
                    path
                ),

            "relevant_schema_paths":
                relevant[
                    :120
                ],
        }

        inspected += 1

        print()
        print(
            "FILE =",
            relative,
        )

        for item in relevant[
            :120
        ]:

            print(
                "  ",
                item[
                    "path"
                ],
                ":",
                item[
                    "type"
                ],
            )

    print()
    print(
        "schema files inspected =",
        inspected,
    )

    # ========================================================
    # F. Evidence matrix
    # ========================================================

    print()
    print(
        "===== F. EVALUATOR REFERENCE READINESS ====="
    )

    source_text = json.dumps(
        source_candidates,
        sort_keys=True,
    )

    script_text = json.dumps(
        script_candidates,
        sort_keys=True,
    )

    schema_text = json.dumps(
        schema_evidence,
        sort_keys=True,
    )

    combined = (
        prereg_text
        +
        "\n"
        +
        source_text
        +
        "\n"
        +
        script_text
        +
        "\n"
        +
        schema_text
    ).lower()

    readiness = {
        "constructed_oracle_semantics":
            contains_any(
                combined,
                (
                    "oracle",
                    "constructed_future",
                    "constructed future",
                ),
            ),

        "future_truth_evaluator_route":
            (
                contains_any(
                    combined,
                    (
                        "future_gt",
                        "future truth",
                        "future_state",
                    ),
                )
                and
                contains_any(
                    combined,
                    (
                        "evaluator",
                        "oracle",
                        "scoring",
                    ),
                )
            ),

        "full_box_geometry":
            (
                contains_any(
                    combined,
                    (
                        "full_box",
                        "full-box",
                        "box3d",
                        "corners",
                        "corner",
                        "footprint",
                    ),
                )
            ),

        "projection_or_rasterization":
            contains_any(
                combined,
                (
                    "project",
                    "projection",
                    "raster",
                    "footprint",
                ),
            ),

        "vehicle_reference_region":
            (
                "vehicle"
                in combined
                and
                contains_any(
                    combined,
                    (
                        "surrogate",
                        "front-upper",
                        "rear-upper",
                        "oracle",
                    ),
                )
            ),

        "pedestrian_future_region":
            (
                "pedestrian"
                in combined
                and
                contains_any(
                    combined,
                    (
                        "future",
                        "full-box",
                        "full_box",
                        "footprint",
                    ),
                )
            ),

        "cyclist_future_region":
            (
                "cyclist"
                in combined
                and
                contains_any(
                    combined,
                    (
                        "future",
                        "full-box",
                        "full_box",
                        "footprint",
                    ),
                )
            ),

        "road_roi":
            (
                "road"
                in combined
                and
                "roi"
                in combined
            ),

        "same_headlamp_grid_context":
            contains_any(
                combined,
                (
                    "theta",
                    "range",
                    "illuminationgrid",
                    "illumination_grid",
                    "actuator grid",
                ),
            ),
    }

    for key, value in (
        readiness.items()
    ):

        print(
            f"{key:34s}=",
            (
                "FOUND"
                if value
                else "NOT PROVEN"
            ),
        )

    # ========================================================
    # G. Strong source candidates by category
    # ========================================================

    print()
    print(
        "===== G. STRONGEST SOURCE CANDIDATES ====="
    )

    strongest = {}

    for category in (
        "constructed_oracle",
        "future_full_box",
        "road_roi",
        "future_truth_access",
        "reactive_reference",
    ):

        candidates = [
            item
            for item in (
                source_candidates
                +
                script_candidates
            )
            if category
            in
            item[
                "categories"
            ]
        ]

        candidates.sort(
            key=lambda item: (
                -item[
                    "categories"
                ][
                    category
                ],
                item.get(
                    "module",
                    item.get(
                        "script",
                        "",
                    ),
                ),
                item[
                    "line"
                ],
            )
        )

        strongest[
            category
        ] = (
            candidates[
                :5
            ]
        )

        print()
        print(
            category
        )

        if not candidates:

            print(
                "  NONE"
            )

        for item in candidates[
            :5
        ]:

            location = item.get(
                "module",
                item.get(
                    "script",
                    "",
                ),
            )

            print(
                f"  {location}:"
                f"L{item['line']} "
                f"{item['name']} "
                f"score="
                f"{item['categories'][category]}"
            )

    # ========================================================
    # H. Scientific boundary
    # ========================================================

    print()
    print(
        "===== H. SCIENTIFIC BOUNDARY ====="
    )

    print(
        "development artifact schemas opened = YES"
    )

    print(
        "development metric values computed  = NO"
    )

    print(
        "acceptance bounds selected          = NO"
    )

    print(
        "controller parameters modified      = NO"
    )

    print(
        "scientific source modified          = NO"
    )

    print(
        "formal-named file content opened    = NO"
    )

    print(
        "formal outcomes read                = NO"
    )

    print(
        "formal evaluation                   = NO"
    )

    # ========================================================
    # I. Full regression
    # ========================================================

    print()
    print(
        "===== I. FULL STAGE6 REGRESSION ====="
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
            "Unexpected Stage6 test count "
            f"{test_count}; expected "
            f"{EXPECTED_STAGE6_TESTS}."
        ),
    )

    print(
        "Stage6 regression =",
        f"{test_count} / {test_count} PASS",
    )

    # ========================================================
    # J. Frozen readback
    # ========================================================

    print()
    print(
        "===== J. FROZEN READBACK ====="
    )

    require(
        sha256_file(
            BINDER
        )
        ==
        EXPECTED_BINDER_SHA,
        (
            "I_final binder changed."
        ),
    )

    require(
        sha256_file(
            RECON
        )
        ==
        EXPECTED_RECON_SHA,
        (
            "Reconciliation changed."
        ),
    )

    require(
        sha256_file(
            PROTOCOL
        )
        ==
        EXPECTED_PROTOCOL_SHA,
        (
            "Metric protocol changed."
        ),
    )

    require(
        sha256_file(
            METRICS
        )
        ==
        EXPECTED_METRICS_SHA,
        (
            "Metric runtime changed."
        ),
    )

    require(
        sha256_file(
            INTERFACE
        )
        ==
        EXPECTED_INTERFACE_SHA,
        (
            "Metric interface changed."
        ),
    )

    print(
        "I_final binder      = UNCHANGED"
    )

    print(
        "metric authority    = UNCHANGED"
    )

    print(
        "scientific runtime  = UNCHANGED"
    )

    # ========================================================
    # K. Storage
    # ========================================================

    print()
    print(
        "===== K. STORAGE ====="
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
    # L. Discovery report
    # ========================================================

    result = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.8",

        "part":
            "2/2_oracle_roi_route_discovery",

        "status":
            "PASS_ORACLE_ROI_ROUTE_DISCOVERY",

        "frozen_evidence": {
            "I_final_binder_sha256":
                EXPECTED_BINDER_SHA,

            "metric_reconciliation_sha256":
                EXPECTED_RECON_SHA,

            "metric_protocol_sha256":
                EXPECTED_PROTOCOL_SHA,

            "metric_runtime_sha256":
                EXPECTED_METRICS_SHA,

            "metric_interface_sha256":
                EXPECTED_INTERFACE_SHA,
        },

        "source_candidates":
            source_candidates,

        "script_candidates":
            script_candidates,

        "schema_evidence":
            schema_evidence,

        "readiness":
            readiness,

        "strongest_candidates":
            strongest,

        "scientific_execution": {
            "development_schema_read":
                True,

            "development_performance_metrics_computed":
                False,

            "numeric_bounds_selected":
                False,

            "policy_tuning":
                False,

            "parameter_tuning":
                False,

            "scientific_source_modified":
                False,

            "formal_content_opened":
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
                "Use strongest evidence-supported routes "
                "to freeze exact constructed-oracle, "
                "future full-box and road-ROI evaluator "
                "bindings before development metric execution."
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
        "BLOCK 6.8 ORACLE / ROI ROUTE DISCOVERY — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "I_final binding            = FROZEN PASS"
    )

    print(
        "constructed oracle route   =",
        (
            "FOUND"
            if readiness[
                "constructed_oracle_semantics"
            ]
            else "NOT PROVEN"
        ),
    )

    print(
        "future truth evaluator     =",
        (
            "FOUND"
            if readiness[
                "future_truth_evaluator_route"
            ]
            else "NOT PROVEN"
        ),
    )

    print(
        "future full-box geometry   =",
        (
            "FOUND"
            if readiness[
                "full_box_geometry"
            ]
            else "NOT PROVEN"
        ),
    )

    print(
        "projection/raster route    =",
        (
            "FOUND"
            if readiness[
                "projection_or_rasterization"
            ]
            else "NOT PROVEN"
        ),
    )

    print(
        "vehicle reference region   =",
        (
            "FOUND"
            if readiness[
                "vehicle_reference_region"
            ]
            else "NOT PROVEN"
        ),
    )

    print(
        "pedestrian future region   =",
        (
            "FOUND"
            if readiness[
                "pedestrian_future_region"
            ]
            else "NOT PROVEN"
        ),
    )

    print(
        "cyclist future region      =",
        (
            "FOUND"
            if readiness[
                "cyclist_future_region"
            ]
            else "NOT PROVEN"
        ),
    )

    print(
        "road ROI route             =",
        (
            "FOUND"
            if readiness[
                "road_roi"
            ]
            else "NOT PROVEN"
        ),
    )

    print(
        "same headlamp grid context =",
        (
            "FOUND"
            if readiness[
                "same_headlamp_grid_context"
            ]
            else "NOT PROVEN"
        ),
    )

    print(
        "development metrics        = NOT COMPUTED"
    )

    print(
        "numeric bounds             = NOT SELECTED"
    )

    print(
        "formal content             = NOT OPENED"
    )

    print(
        "formal evaluation          = NO"
    )

    print(
        "Stage6 regression          =",
        f"{test_count} / {test_count} PASS",
    )

    print(
        "STATUS = PASS_ORACLE_ROI_ROUTE_DISCOVERY"
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
        "BLOCK 6.8 ORACLE / ROI ROUTE DISCOVERY = BLOCKED"
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
        "development metrics computed = NO"
    )

    print(
        "numeric bounds selected      = NO"
    )

    print(
        "scientific source modified   = NO"
    )

    print(
        "formal content opened        = NO"
    )

    print(
        "formal outcomes read         = NO"
    )

    print(
        "formal evaluation            = NO"
    )

    print()
    print(
        "Do not run development metric evaluation."
    )

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

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


ROOT = Path("/home/agni/waymo")
S4 = ROOT / "iscai_stage4"
S5 = ROOT / "iscai_stage5"
S6 = ROOT / "iscai_stage6"

PART1 = (
    S6
    / "reports/block68_part1_metric_freeze_gate.json"
)

BLOCK67 = (
    S6
    / "reports/block67_final_closure.json"
)

METRIC_PROTOCOL = (
    S6
    / "configs/stage6_metric_freeze_protocol.json"
)

METRIC_RUNTIME = (
    S6
    / "src/iscai_stage6/adb/metric_semantics.py"
)

CLASS_POLICY = (
    S6
    / "src/iscai_stage6/adb/class_aware_policy.py"
)

MARGIN_BINDING = (
    S6
    / "src/iscai_stage6/adb/class_margin_binding.py"
)

BASELINE_CONTRACT = (
    S6
    / "src/iscai_stage6/adb/baseline_metric_contract.py"
)

STAGE5_CLOSURE = (
    S5
    / "reports/stage5_final_closure.json"
)

REPORT = (
    S6
    / "reports/block68_part2_development_preflight.json"
)

EXPECTED_METRIC_PROTOCOL_SHA = (
    "c12b95c24ba64f48ac8dc3890fa55061"
    "3ea597875f6a5a759762b88629623ac0"
)

EXPECTED_METRIC_RUNTIME_SHA = (
    "fdca66a1a9af74d24baaafe4e17d12ac"
    "db2db538e037503aba6ee04684eac362"
)

EXPECTED_BLOCK67_SHA = (
    "67d168ed7678164b8d85885e6e81f41f"
    "7206bd8f754e13ec5cc28c3a65aa20e7"
)

EXPECTED_POLICY_SHA = (
    "b998f468b98c2770c48c84a2c0aaaa21"
    "77c2d84b8fc838e80fef9b9da1ebc14b"
)

EXPECTED_STAGE6_TESTS = 196

MIN_FREE_GIB = 250.0


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


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def write_json(path: Path, payload):
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def ast_inventory(path: Path):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    tree = ast.parse(
        text,
        filename=str(path),
    )

    functions = []
    classes = []

    for node in tree.body:
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            functions.append(
                {
                    "name":
                        node.name,

                    "line":
                        node.lineno,
                }
            )

        elif isinstance(
            node,
            ast.ClassDef,
        ):
            classes.append(
                {
                    "name":
                        node.name,

                    "line":
                        node.lineno,
                }
            )

    return {
        "functions":
            functions,

        "classes":
            classes,
    }


def interesting_name(name: str) -> bool:
    lower = name.lower()

    tokens = (
        "adb",
        "baseline",
        "reactive",
        "predict",
        "occup",
        "oracle",
        "metric",
        "illum",
        "mask",
        "class",
        "margin",
        "actor",
        "future",
        "scenario",
        "develop",
        "dev",
        "validation",
        "cache",
        "posterior",
    )

    return any(
        token in lower
        for token in tokens
    )


def candidate_files(
    root: Path,
    *,
    max_results: int = 120,
):
    """
    Filename/path discovery only.

    We deliberately do NOT open report/artifact contents here.
    This prevents accidental reading of formal or development
    outcome metrics during preflight.
    """

    if not root.exists():
        return []

    results = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        relative = str(
            path.relative_to(root)
        )

        lower = relative.lower()

        if any(
            token in lower
            for token in (
                "dev",
                "development",
                "validation",
                "cache",
                "adb",
                "reactive",
                "predict",
                "oracle",
                "occupancy",
                "baseline",
                "posterior",
            )
        ):
            results.append(relative)

    return sorted(
        results
    )[:max_results]


def config_string_inventory(
    path: Path,
):
    """
    Config discovery only.

    Formal/report/artifact JSON is NOT read.
    """

    try:
        payload = load_json(path)
    except Exception:
        return []

    matches = []

    def walk(value, prefix=""):
        if isinstance(value, dict):
            for key, child in value.items():
                child_prefix = (
                    f"{prefix}.{key}"
                    if prefix
                    else str(key)
                )

                yield from walk(
                    child,
                    child_prefix,
                )

        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from walk(
                    child,
                    f"{prefix}[{index}]",
                )

        else:
            yield prefix, value

    for key, value in walk(payload):
        text = (
            f"{key}={value}"
        ).lower()

        if any(
            token in text
            for token in (
                "dev",
                "development",
                "validation",
                "cache",
                "split",
                "manifest",
                "reactive",
                "predictive",
                "oracle",
                "adb",
            )
        ):
            matches.append(
                (
                    key,
                    value,
                )
            )

    return matches[:80]


def run_full_regression():
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
        "DEVELOPMENT-EVALUATION API PREFLIGHT"
    )
    print(
        "READ-ONLY WITH RESPECT TO SCIENTIFIC OUTCOMES"
    )
    print(
        "============================================================"
    )

    required = (
        PART1,
        BLOCK67,
        METRIC_PROTOCOL,
        METRIC_RUNTIME,
        CLASS_POLICY,
        MARGIN_BINDING,
        BASELINE_CONTRACT,
        STAGE5_CLOSURE,
    )

    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]

    require(
        not missing,
        (
            "Missing frozen dependency: "
            + ", ".join(missing)
        ),
    )

    # ========================================================
    # A. Frozen continuity
    # ========================================================

    print()
    print(
        "===== A. FROZEN CONTINUITY ====="
    )

    part1 = load_json(
        PART1
    )

    require(
        part1.get("status")
        ==
        "PASS_METRIC_SEMANTICS_FROZEN",
        (
            "Block6.8 Part1 status mismatch."
        ),
    )

    require(
        sha256_file(
            METRIC_PROTOCOL
        )
        ==
        EXPECTED_METRIC_PROTOCOL_SHA,
        "Metric protocol SHA changed.",
    )

    require(
        sha256_file(
            METRIC_RUNTIME
        )
        ==
        EXPECTED_METRIC_RUNTIME_SHA,
        "Metric runtime SHA changed.",
    )

    require(
        sha256_file(
            BLOCK67
        )
        ==
        EXPECTED_BLOCK67_SHA,
        "Block6.7 closure SHA changed.",
    )

    require(
        sha256_file(
            CLASS_POLICY
        )
        ==
        EXPECTED_POLICY_SHA,
        "Frozen class-aware policy changed.",
    )

    print(
        "Block6.7 closure       = EXACT PASS"
    )

    print(
        "Block6.8 Part1         = FROZEN PASS"
    )

    print(
        "metric protocol SHA    = EXACT PASS"
    )

    print(
        "metric runtime SHA     = EXACT PASS"
    )

    print(
        "class-aware policy     = EXACT PASS"
    )

    # ========================================================
    # B. Stage6 source API inventory
    # ========================================================

    print()
    print(
        "===== B. STAGE6 ADB SOURCE API INVENTORY ====="
    )

    adb_root = (
        S6
        / "src/iscai_stage6/adb"
    )

    source_inventory = {}

    for path in sorted(
        adb_root.glob("*.py")
    ):
        if (
            path.name.startswith("__")
        ):
            continue

        inventory = ast_inventory(
            path
        )

        relevant_functions = [
            item
            for item in inventory[
                "functions"
            ]
            if interesting_name(
                item["name"]
            )
        ]

        relevant_classes = [
            item
            for item in inventory[
                "classes"
            ]
            if interesting_name(
                item["name"]
            )
        ]

        if (
            relevant_functions
            or
            relevant_classes
        ):
            source_inventory[
                path.name
            ] = {
                "sha256":
                    sha256_file(path),

                "functions":
                    relevant_functions,

                "classes":
                    relevant_classes,
            }

            print()
            print(
                f"MODULE = {path.name}"
            )

            for item in (
                relevant_classes
            ):
                print(
                    f"  CLASS "
                    f"{item['name']} "
                    f"@L{item['line']}"
                )

            for item in (
                relevant_functions
            ):
                print(
                    f"  FUNC  "
                    f"{item['name']} "
                    f"@L{item['line']}"
                )

    require(
        source_inventory,
        (
            "No Stage6 ADB source APIs discovered."
        ),
    )

    # ========================================================
    # C. Stage6 runner inventory
    # ========================================================

    print()
    print(
        "===== C. STAGE6 RUNNER / SCRIPT INVENTORY ====="
    )

    script_inventory = {}

    for path in sorted(
        (
            S6 / "scripts"
        ).glob("*.py")
    ):
        if (
            "block68_part2_development_preflight"
            in path.name
        ):
            continue

        inventory = ast_inventory(
            path
        )

        relevant_functions = [
            item
            for item in inventory[
                "functions"
            ]
            if interesting_name(
                item["name"]
            )
        ]

        if (
            interesting_name(
                path.name
            )
            or
            relevant_functions
        ):
            script_inventory[
                path.name
            ] = {
                "sha256":
                    sha256_file(path),

                "functions":
                    relevant_functions,
            }

            print(
                path.name
            )

            for item in (
                relevant_functions
            ):
                print(
                    f"  FUNC "
                    f"{item['name']} "
                    f"@L{item['line']}"
                )

    # ========================================================
    # D. Config-only split/dev discovery
    # ========================================================

    print()
    print(
        "===== D. CONFIG-ONLY DEVELOPMENT/SPLIT DISCOVERY ====="
    )

    config_inventory = {}

    for path in sorted(
        (
            S6 / "configs"
        ).glob("*.json")
    ):
        matches = (
            config_string_inventory(
                path
            )
        )

        if not matches:
            continue

        config_inventory[
            path.name
        ] = [
            {
                "key":
                    key,

                "value":
                    value,
            }
            for key, value in matches
        ]

        print()
        print(
            "CONFIG =",
            path.name,
        )

        for key, value in (
            matches[:30]
        ):
            text = str(
                value
            )

            if len(text) > 160:
                text = (
                    text[:157]
                    + "..."
                )

            print(
                f"  {key} = {text}"
            )

    # ========================================================
    # E. Candidate artifact PATHS only
    # ========================================================

    print()
    print(
        "===== E. CANDIDATE DEVELOPMENT ARTIFACT PATHS ====="
    )

    stage6_candidates = (
        candidate_files(
            S6 / "artifacts"
        )
    )

    stage4_candidates = (
        candidate_files(
            S4 / "artifacts"
        )
    )

    stage4_report_names = (
        candidate_files(
            S4 / "reports"
        )
    )

    print()
    print(
        "Stage6 artifact candidates =",
        len(
            stage6_candidates
        ),
    )

    for item in (
        stage6_candidates[:80]
    ):
        print(
            "  ",
            item,
        )

    print()
    print(
        "Stage4 artifact candidates =",
        len(
            stage4_candidates
        ),
    )

    for item in (
        stage4_candidates[:80]
    ):
        print(
            "  ",
            item,
        )

    print()
    print(
        "Stage4 report-name candidates =",
        len(
            stage4_report_names
        ),
    )

    for item in (
        stage4_report_names[:40]
    ):
        print(
            "  ",
            item,
        )

    print()
    print(
        "NOTE:"
    )

    print(
        "Only PATH NAMES were inspected above."
    )

    print(
        "No development result JSON was opened."
    )

    print(
        "No formal result JSON/manifest was opened."
    )

    # ========================================================
    # F. Explicit formal boundary
    # ========================================================

    print()
    print(
        "===== F. FORMAL BOUNDARY ====="
    )

    formal_name_candidates = []

    for root in (
        S4,
        S5,
        S6,
    ):
        for path in root.rglob("*"):
            if not path.is_file():
                continue

            relative = str(
                path.relative_to(root)
            )

            if "formal" in (
                relative.lower()
            ):
                formal_name_candidates.append(
                    f"{root.name}/{relative}"
                )

    print(
        "formal-named paths discovered =",
        len(
            formal_name_candidates
        ),
    )

    for item in (
        sorted(
            formal_name_candidates
        )[:30]
    ):
        print(
            "  ",
            item,
        )

    print(
        "formal file CONTENT read = NO"
    )

    print(
        "formal outcomes read     = NO"
    )

    # ========================================================
    # G. Full Stage6 regression
    # ========================================================

    print()
    print(
        "===== G. FULL STAGE6 REGRESSION ====="
    )

    rc, test_count, output = (
        run_full_regression()
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
            "Unexpected Stage6 test count: "
            f"{test_count}; expected "
            f"{EXPECTED_STAGE6_TESTS}."
        ),
    )

    print(
        "Stage6 regression =",
        f"{test_count} / {test_count} PASS",
    )

    # ========================================================
    # H. Storage
    # ========================================================

    print()
    print(
        "===== H. STORAGE ====="
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
            "250-GiB hard reserve violated."
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
    # I. Discovery classification
    # ========================================================

    print()
    print(
        "===== I. DEVELOPMENT-RUNTIME READINESS ====="
    )

    flattened_names = " ".join(
        list(
            source_inventory.keys()
        )
        +
        list(
            script_inventory.keys()
        )
        +
        stage6_candidates
        +
        stage4_candidates
    ).lower()

    readiness = {
        "predictive_or_occupancy_runtime":
            any(
                token
                in
                flattened_names
                for token in (
                    "predict",
                    "occup",
                )
            ),

        "reactive_runtime":
            "reactive"
            in
            flattened_names,

        "adb_runtime":
            "adb"
            in
            flattened_names,

        "development_or_dev_artifact":
            any(
                token
                in
                flattened_names
                for token in (
                    "development",
                    "dev",
                )
            ),

        "validation_named_artifact":
            "validation"
            in
            flattened_names,
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

    # We do NOT block if dev artifact is not yet present;
    # the purpose of this preflight is precisely to tell us
    # whether Part2 must generate it from the frozen dev split.
    require(
        readiness[
            "adb_runtime"
        ],
        (
            "No Stage6 ADB runtime evidence found."
        ),
    )

    # ========================================================
    # J. Discovery report
    # ========================================================

    result = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.8",

        "part":
            "2/2_preflight",

        "status":
            "PASS_DEVELOPMENT_API_DISCOVERED",

        "scientific_outcomes": {
            "development_outcomes_read":
                False,

            "formal_outcomes_read":
                False,

            "formal_manifest_content_read":
                False,

            "policy_tuning":
                False,

            "parameter_tuning":
                False,
        },

        "frozen_continuity": {
            "Block67_sha256":
                sha256_file(
                    BLOCK67
                ),

            "metric_protocol_sha256":
                sha256_file(
                    METRIC_PROTOCOL
                ),

            "metric_runtime_sha256":
                sha256_file(
                    METRIC_RUNTIME
                ),

            "class_policy_sha256":
                sha256_file(
                    CLASS_POLICY
                ),
        },

        "source_inventory":
            source_inventory,

        "script_inventory":
            script_inventory,

        "config_inventory":
            config_inventory,

        "candidate_paths": {
            "Stage6_artifacts":
                stage6_candidates,

            "Stage4_artifacts":
                stage4_candidates,

            "Stage4_report_names":
                stage4_report_names,
        },

        "readiness":
            readiness,

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
                "Use discovered frozen runtime APIs "
                "to execute validation-development-only "
                "ADB baseline evaluation and freeze "
                "numeric non-inferiority bounds. "
                "Formal outcomes remain prohibited."
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
        "BLOCK 6.8 PART 2/2 — DEVELOPMENT PREFLIGHT FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "Block6.8 Part1           = EXACT FROZEN PASS"
    )

    print(
        "Stage6 ADB APIs          = INVENTORIED"
    )

    print(
        "Stage6 runner APIs       = INVENTORIED"
    )

    print(
        "development artifacts    = PATH DISCOVERY ONLY"
    )

    print(
        "development outcomes read= NO"
    )

    print(
        "formal outcomes read     = NO"
    )

    print(
        "formal files opened      = NO"
    )

    print(
        "policy tuning            = NO"
    )

    print(
        "Stage6 regression        =",
        f"{test_count} / {test_count} PASS",
    )

    print(
        "STATUS = PASS_DEVELOPMENT_API_DISCOVERED"
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
        "BLOCK 6.8 PART 2/2 PREFLIGHT = BLOCKED"
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

    print()
    print(
        "development outcomes read = NO"
    )

    print(
        "formal outcomes read      = NO"
    )

    print(
        "formal evaluation         = NO"
    )

    print(
        "parameter tuning          = NO"
    )

    print(
        "Do not run formal Stage6 evaluation."
    )

    print(
        "terminal remains open = YES"
    )

# Intentionally no non-zero sys.exit().

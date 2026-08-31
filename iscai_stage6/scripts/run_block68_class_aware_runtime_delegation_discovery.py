from __future__ import annotations

import ast
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

BINDER = (
    S6
    / "scripts/"
      "run_block68_class_aware_frozen_policy_binding.py"
)

PART3C1 = (
    S6
    / "scripts/"
      "run_block66_part3c1_class_aware_runtime.py"
)

HANDOFF = (
    S6
    / "artifacts/block66/"
      "block66_part3c1_to_part3c2_handoff.json"
)

FINAL_FREEZE = (
    S6
    / "artifacts/block66/"
      "block66_part3c_final_freeze_manifest.json"
)

PREREG = (
    S6
    / "configs/"
      "block66_part3b_class_aware_policy_preregistration.json"
)

NI_DELTA = (
    S6
    / "configs/"
      "stage6_exact_noninferiority_deltas.json"
)

ACCEPTANCE = (
    S6
    / "configs/"
      "stage6_preformal_primary_acceptance_policy.json"
)

TIMESTAMP = (
    S6
    / "configs/"
      "stage6_reactive_future_truth_timestamp_alignment_superseding.json"
)

REPORT = (
    S6
    / "reports/"
      "block68_class_aware_runtime_delegation_discovery.json"
)


EXPECTED_BINDER_SHA = (
    "41086d10733b218f3fef1ec1e3e6d1e7"
    "98fb87e0c6e75c82ef6e190819638822"
)

EXPECTED_PREREG_SHA = (
    "5704494a89b4b3c7a3c19b0df8176c55"
    "0c00795ed17cf2906c343f340461429e"
)

EXPECTED_NI_SHA = (
    "a77e548e060c314dd98a7220b0f9ed3a"
    "ebd5d1978eb70f456b82920e792a29d4"
)

EXPECTED_ACCEPTANCE_SHA = (
    "e103466b1a6eb62be4e6746029147cbda"
    "9671b05130acba5210a3ab7328e6b81"
)

EXPECTED_TIMESTAMP_SHA = (
    "51a3b56a119467001a9f4617e1e19dc8"
    "2c548c0a30c79fc9c4a9c1f5ef27bf56"
)

EXPECTED_TESTS = 224


RUNTIME_SYMBOLS = (
    "threshold_occupancy_counts_strict_k",
    "threshold_actor_occupancy",
    "compute_class_aware_margin",
    "bind_predictive_class_margin",
    "angular_margin_cells",
    "dilate_mask_theta",
    "predictive_mask_to_illumination",
    "compose_class_aware_illumination",
    "temporal_smooth_schedule",
    "apply_actuation_rate_limit",
)

DISCOVERY_TOKENS = (
    "selected",
    "selection",
    "winner",
    "winning",
    "best",
    "policy",
    "gamma",
    "threshold",
    "floor",
    "margin",
    "time_constant",
    "smoothing",
    "rho_dim",
    "rho_bright",
    "rate_limit",
    "class_aware",
    "part3c",
)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def file_sha(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
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
        cwd=str(S6),
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
            count = int(match.group(1))

    require(
        process.returncode == 0,
        (
            "Stage6 regression failed:\n"
            +
            "\n".join(
                process.stdout.splitlines()[-80:]
            )
        ),
    )

    require(
        count == EXPECTED_TESTS,
        (
            f"Expected {EXPECTED_TESTS} tests; "
            f"got {count}."
        ),
    )

    return count


def schema_paths(value, prefix=""):
    """
    Return JSON key structure only.
    Scalar scientific values are deliberately not included.
    """

    paths = []

    if isinstance(value, dict):

        for key in sorted(value):

            path = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            paths.append(path)

            paths.extend(
                schema_paths(
                    value[key],
                    path,
                )
            )

    elif isinstance(value, list):

        path = (
            f"{prefix}[]"
        )

        paths.append(path)

        # Schema discovery only.
        for item in value[:1]:
            paths.extend(
                schema_paths(
                    item,
                    path,
                )
            )

    return paths


def pathlike_values(value, prefix=""):
    """
    Extract only references to files/modules.
    Do not print numerical scientific outcomes.
    """

    rows = []

    if isinstance(value, dict):

        for key, child in value.items():

            path = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            rows.extend(
                pathlike_values(
                    child,
                    path,
                )
            )

    elif isinstance(value, list):

        for index, child in enumerate(value):

            rows.extend(
                pathlike_values(
                    child,
                    f"{prefix}[{index}]",
                )
            )

    elif isinstance(value, str):

        lower = value.lower()

        if any(
            extension in lower
            for extension in (
                ".json",
                ".jsonl",
                ".npy",
                ".npz",
                ".py",
                ".pt",
                ".pth",
            )
        ):
            rows.append(
                {
                    "key_path":
                        prefix,

                    "value":
                        value,
                }
            )

    return rows


def import_inventory(path: Path):
    source = path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(path),
    )

    rows = []

    for node in ast.walk(tree):

        if isinstance(node, ast.Import):

            for alias in node.names:
                rows.append(
                    {
                        "line":
                            node.lineno,

                        "kind":
                            "import",

                        "module":
                            alias.name,

                        "name":
                            None,

                        "asname":
                            alias.asname,
                    }
                )

        elif isinstance(node, ast.ImportFrom):

            for alias in node.names:
                rows.append(
                    {
                        "line":
                            node.lineno,

                        "kind":
                            "from",

                        "module":
                            node.module,

                        "level":
                            node.level,

                        "name":
                            alias.name,

                        "asname":
                            alias.asname,
                    }
                )

    rows.sort(
        key=lambda item: (
            item["line"],
            str(item["module"]),
            str(item["name"]),
        )
    )

    return rows


def call_inventory(path: Path):
    source = path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(path),
    )

    rows = []

    for node in ast.walk(tree):

        if not isinstance(node, ast.Call):
            continue

        try:
            expression = ast.unparse(
                node.func
            )
        except Exception:
            continue

        rows.append(
            {
                "line":
                    int(node.lineno),

                "call":
                    expression,
            }
        )

    rows.sort(
        key=lambda item:
            item["line"]
    )

    return rows


def interesting_source_context(path: Path):
    lines = path.read_text(
        encoding="utf-8"
    ).splitlines()

    matching = set()

    for index, line in enumerate(
        lines,
        start=1,
    ):
        lower = line.lower()

        if any(
            token in lower
            for token in DISCOVERY_TOKENS
        ):
            for line_number in range(
                max(1, index - 2),
                min(
                    len(lines),
                    index + 2,
                )
                +
                1,
            ):
                matching.add(
                    line_number
                )

    return [
        {
            "line":
                line_number,

            "text":
                lines[
                    line_number - 1
                ],
        }
        for line_number
        in sorted(matching)
    ]


def scan_python_runtime_symbols():
    roots = (
        S6 / "src/iscai_stage6",
        S6 / "scripts",
    )

    hits = []

    for root in roots:

        if not root.is_dir():
            continue

        for path in sorted(
            root.rglob("*.py")
        ):

            lower_path = str(path).lower()

            # Never inspect Stage6 formal-evaluation scripts.
            if (
                "block69" in lower_path
                or
                "formal_evaluation" in lower_path
                or
                "formal_eval" in lower_path
            ):
                continue

            source = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

            lines = source.splitlines()

            for symbol in RUNTIME_SYMBOLS:

                for line_number, line in enumerate(
                    lines,
                    start=1,
                ):

                    if symbol not in line:
                        continue

                    hits.append(
                        {
                            "path":
                                str(
                                    path.relative_to(
                                        S6
                                    )
                                ),

                            "sha256":
                                file_sha(path),

                            "line":
                                line_number,

                            "symbol":
                                symbol,

                            "text":
                                line.strip(),
                        }
                    )

    return hits


def candidate_filenames():
    """
    Filename inventory only; files are NOT opened.
    """

    roots = (
        S6 / "configs",
        S6 / "artifacts/block66",
        S6 / "reports",
    )

    rows = []

    filename_tokens = (
        "policy",
        "winner",
        "winning",
        "selected",
        "selection",
        "part3c",
        "gamma",
        "floor",
        "margin",
        "temporal",
        "rate",
        "freeze",
        "handoff",
    )

    for root in roots:

        if not root.is_dir():
            continue

        for path in sorted(
            root.iterdir()
        ):

            if not path.is_file():
                continue

            lower = path.name.lower()

            if (
                "formal_evaluation" in lower
                or
                "block69" in lower
            ):
                continue

            if not any(
                token in lower
                for token in filename_tokens
            ):
                continue

            rows.append(
                {
                    "path":
                        str(
                            path.relative_to(
                                S6
                            )
                        ),

                    "size_bytes":
                        path.stat().st_size,

                    "sha256":
                        file_sha(path),
                }
            )

    return rows


def main():
    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.8"
    )
    print(
        "CLASS-AWARE RUNTIME DELEGATION DISCOVERY"
    )
    print(
        "SOURCE + SCHEMA ONLY"
    )
    print(
        "NO P_OCC / NO PREDICTIVE PERFORMANCE / NO FORMAL"
    )
    print(
        "============================================================"
    )

    # ========================================================
    # A. Exact current boundary
    # ========================================================

    print()
    print(
        "===== A. EXACT CURRENT BOUNDARY ====="
    )

    require(
        file_sha(
            BINDER
        )
        ==
        EXPECTED_BINDER_SHA,
        (
            "Policy-binding runner differs from "
            "exact preformal-name-gate repaired SHA."
        ),
    )

    require(
        file_sha(
            PREREG
        )
        ==
        EXPECTED_PREREG_SHA,
        (
            "Block6.6 preregistration changed."
        ),
    )

    require(
        file_sha(
            NI_DELTA
        )
        ==
        EXPECTED_NI_SHA,
        (
            "NI delta freeze changed."
        ),
    )

    require(
        file_sha(
            ACCEPTANCE
        )
        ==
        EXPECTED_ACCEPTANCE_SHA,
        (
            "Acceptance policy changed."
        ),
    )

    require(
        file_sha(
            TIMESTAMP
        )
        ==
        EXPECTED_TIMESTAMP_SHA,
        (
            "Timestamp alignment changed."
        ),
    )

    for path in (
        PART3C1,
        HANDOFF,
        FINAL_FREEZE,
    ):
        require(
            path.is_file(),
            f"Required Block66 file missing: {path}",
        )

    print(
        "binding runner          = EXACT PASS"
    )
    print(
        "Block6.6 prereg         = EXACT PASS"
    )
    print(
        "NI bounds               = EXACT PASS"
    )
    print(
        "acceptance policy       = EXACT PASS"
    )
    print(
        "timestamp alignment     = EXACT PASS"
    )
    print(
        "P_occ values            = NOT OPENED"
    )
    print(
        "predictive performance  = NOT COMPUTED"
    )
    print(
        "formal content          = NOT OPENED"
    )

    # ========================================================
    # B. Part3C1 actual source structure
    # ========================================================

    print()
    print(
        "===== B. PART3C1 SOURCE / IMPORT STRUCTURE ====="
    )

    print(
        "Part3C1 path   =",
        PART3C1,
    )

    print(
        "Part3C1 SHA256 =",
        file_sha(
            PART3C1
        ),
    )

    imports = import_inventory(
        PART3C1
    )

    calls = call_inventory(
        PART3C1
    )

    print()
    print(
        "imports:"
    )

    for item in imports:
        print(
            " ",
            json.dumps(
                item,
                sort_keys=True,
            )
        )

    print()
    print(
        "all call expressions:"
    )

    for item in calls:
        print(
            f"  L{item['line']:05d}",
            item[
                "call"
            ],
        )

    # ========================================================
    # C. Relevant Part3C1 source context
    # ========================================================

    print()
    print(
        "===== C. PART3C1 POLICY/SELECTION SOURCE CONTEXT ====="
    )

    context = interesting_source_context(
        PART3C1
    )

    for item in context:
        print(
            f"L{item['line']:05d}: "
            f"{item['text']}"
        )

    # ========================================================
    # D. Handoff/freeze SCHEMA ONLY
    # ========================================================

    print()
    print(
        "===== D. HANDOFF + FINAL-FREEZE SCHEMA ====="
    )

    handoff = read_json(
        HANDOFF
    )

    freeze = read_json(
        FINAL_FREEZE
    )

    handoff_schema = schema_paths(
        handoff
    )

    freeze_schema = schema_paths(
        freeze
    )

    print()
    print(
        "PART3C1 HANDOFF key paths:"
    )

    for path in handoff_schema:
        print(
            " ",
            path,
        )

    print()
    print(
        "PART3C FINAL FREEZE key paths:"
    )

    for path in freeze_schema:
        print(
            " ",
            path,
        )

    # ========================================================
    # E. Referenced artifact paths only
    # ========================================================

    print()
    print(
        "===== E. REFERENCED FILE PATHS FROM FREEZES ====="
    )

    handoff_paths = pathlike_values(
        handoff
    )

    freeze_paths = pathlike_values(
        freeze
    )

    print()
    print(
        "handoff path-like values:"
    )

    for item in handoff_paths:
        print(
            " ",
            item[
                "key_path"
            ],
            "=",
            item[
                "value"
            ],
        )

    print()
    print(
        "final-freeze path-like values:"
    )

    for item in freeze_paths:
        print(
            " ",
            item[
                "key_path"
            ],
            "=",
            item[
                "value"
            ],
        )

    # ========================================================
    # F. Runtime symbols anywhere in non-formal Stage6 source
    # ========================================================

    print()
    print(
        "===== F. RUNTIME SYMBOL LOCATIONS ====="
    )

    symbol_hits = (
        scan_python_runtime_symbols()
    )

    require(
        symbol_hits,
        (
            "No class-aware runtime symbols found "
            "anywhere in non-formal Stage6 source."
        ),
    )

    by_file = {}

    for hit in symbol_hits:
        by_file.setdefault(
            hit[
                "path"
            ],
            [],
        ).append(
            hit
        )

    for path in sorted(by_file):

        print()
        print(
            path,
            "SHA256=",
            by_file[path][0][
                "sha256"
            ],
        )

        for hit in by_file[path]:
            print(
                f"  L{hit['line']:05d}",
                hit[
                    "symbol"
                ],
                "::",
                hit[
                    "text"
                ],
            )

    # ========================================================
    # G. Candidate frozen artifacts — filenames only
    # ========================================================

    print()
    print(
        "===== G. BLOCK66 POLICY/FREEZE CANDIDATE FILES ====="
    )

    candidates = candidate_filenames()

    for item in candidates:
        print(
            item[
                "path"
            ],
            "| bytes=",
            item[
                "size_bytes"
            ],
            "| sha256=",
            item[
                "sha256"
            ],
        )

    # ========================================================
    # H. Boundary / regression
    # ========================================================

    print()
    print(
        "===== H. SCIENTIFIC BOUNDARY ====="
    )

    tests = run_regression()

    print(
        "JSON numeric selection artifacts newly opened = NO"
    )
    print(
        "P_occ manifest content      = NOT OPENED"
    )
    print(
        "P_occ arrays                = NOT OPENED"
    )
    print(
        "predictive illumination     = NOT COMPUTED"
    )
    print(
        "predictive performance      = NOT COMPUTED"
    )
    print(
        "acceptance gate             = NOT TESTED"
    )
    print(
        "policy numerics changed     = NO"
    )
    print(
        "formal evaluation           = NO"
    )
    print(
        "Stage6 regression           =",
        f"{tests} / {tests} PASS",
    )

    # ========================================================
    # I. Report
    # ========================================================

    result = {
        "stage":
            6,

        "block":
            "6.8_class_aware_runtime_delegation_discovery",

        "status":
            "PASS_CLASS_AWARE_RUNTIME_DELEGATION_DISCOVERY_ONLY",

        "part3c1": {
            "path":
                str(
                    PART3C1
                ),

            "sha256":
                file_sha(
                    PART3C1
                ),

            "imports":
                imports,

            "calls":
                calls,

            "policy_selection_source_context":
                context,
        },

        "handoff": {
            "path":
                str(
                    HANDOFF
                ),

            "sha256":
                file_sha(
                    HANDOFF
                ),

            "schema_paths":
                handoff_schema,

            "referenced_paths":
                handoff_paths,
        },

        "final_freeze": {
            "path":
                str(
                    FINAL_FREEZE
                ),

            "sha256":
                file_sha(
                    FINAL_FREEZE
                ),

            "schema_paths":
                freeze_schema,

            "referenced_paths":
                freeze_paths,
        },

        "runtime_symbol_hits":
            symbol_hits,

        "candidate_filenames_only":
            candidates,

        "scientific_boundary": {
            "P_occ_opened":
                False,

            "predictive_performance_computed":
                False,

            "acceptance_tested":
                False,

            "selected_policy_artifact_content_newly_opened":
                False,

            "policy_tuning":
                False,

            "formal_content_opened":
                False,

            "formal_evaluation":
                False,
        },

        "Stage6_regression":
            tests,

        "next":
            (
                "bind exact selected policy from the "
                "specific frozen runtime/helper/artifact "
                "identified by this discovery"
            ),
    }

    REPORT.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 CLASS-AWARE RUNTIME DELEGATION — FINAL"
    )
    print(
        "============================================================"
    )
    print(
        "Part3C1 direct runtime calls = NOT REQUIRED"
    )
    print(
        "Part3C1 import graph         = EXTRACTED"
    )
    print(
        "handoff/freeze schemas       = EXTRACTED"
    )
    print(
        "runtime symbol locations     = EXTRACTED"
    )
    print(
        "candidate policy files       = FILENAMES ONLY"
    )
    print(
        "P_occ values                 = NOT OPENED"
    )
    print(
        "predictive outcomes           = NOT COMPUTED"
    )
    print(
        "acceptance gate               = NOT TESTED"
    )
    print(
        "policy tuning                 = NO"
    )
    print(
        "formal evaluation             = NO"
    )
    print(
        "Stage6 regression             =",
        f"{tests} / {tests} PASS",
    )
    print(
        "STATUS = "
        "PASS_CLASS_AWARE_RUNTIME_DELEGATION_DISCOVERY_ONLY"
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
        "BLOCK 6.8 CLASS-AWARE RUNTIME DELEGATION = BLOCKED"
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
        "P_occ content          = NOT OPENED"
    )
    print(
        "predictive performance = NOT COMPUTED"
    )
    print(
        "acceptance gate        = NOT TESTED"
    )
    print(
        "policy tuning          = NO"
    )
    print(
        "formal evaluation      = NO"
    )
    print(
        "Do not infer selected values "
        "from preregistered candidate spaces."
    )
    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

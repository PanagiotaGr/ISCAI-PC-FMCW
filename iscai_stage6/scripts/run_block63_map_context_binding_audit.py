from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import re
import traceback


ROOT = Path("/home/agni/waymo")
S4 = ROOT / "iscai_stage4"
S6 = ROOT / "iscai_stage6"

BLOCKED_REPORT = (
    S6
    / "reports/block63_part2b1_frozen_gaussian_forward.json"
)

DIAGNOSTIC = (
    S6
    / "artifacts/block63/"
      "block63_part2b1_runtime_boundary_excerpt.txt"
)

REPORT = (
    S6
    / "reports/block63_map_context_binding_audit.json"
)

EXCERPT = (
    S6
    / "artifacts/block63/"
      "block63_map_context_exact_source_excerpt.txt"
)

EXPECTED_BLOCKED_SHA = (
    "fd72d2b2bbadca66251f2026948ff62f"
    "aad7fc84f1201d45ea66f2f0861b6557"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def source_contexts(
    root: Path,
    pattern: re.Pattern,
    *,
    maximum: int = 160,
):
    results = []

    for path in sorted(root.rglob("*.py")):
        try:
            lines = path.read_text(
                encoding="utf-8"
            ).splitlines()

        except BaseException:
            continue

        for index, line in enumerate(lines):
            if not pattern.search(line):
                continue

            lo = max(0, index - 12)
            hi = min(len(lines), index + 13)

            results.append({
                "path": str(path),
                "line": index + 1,
                "context": "\n".join(
                    f"{j + 1:05d}: {lines[j]}"
                    for j in range(lo, hi)
                ),
            })

            if len(results) >= maximum:
                return results

    return results


def exact_definition(path: Path, name: str):
    source = path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(path),
    )

    found = []

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            continue

        if node.name != name:
            continue

        segment = ast.get_source_segment(
            source,
            node,
        )

        if segment is not None:
            found.append({
                "path": str(path),
                "line": node.lineno,
                "source": segment,
            })

    return found


def main():
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.3 — EXACT STAGE4 MAP-CONTEXT BINDING AUDIT"
    )
    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # A. Blocked-state seal
    # --------------------------------------------------------

    print()
    print(
        "===== A. BLOCKED PART2B/1 SEAL ====="
    )

    require(
        BLOCKED_REPORT.is_file(),
        f"Missing {BLOCKED_REPORT}",
    )

    require(
        sha256_file(BLOCKED_REPORT)
        ==
        EXPECTED_BLOCKED_SHA,
        (
            "Part2B/1 blocked report changed."
        ),
    )

    blocked = json.loads(
        BLOCKED_REPORT.read_text(
            encoding="utf-8"
        )
    )

    require(
        blocked.get("status")
        ==
        (
            "BLOCKED_EXACT_FROZEN_"
            "GAUSSIAN_FORWARD_BOUNDARY"
        ),
        (
            "Unexpected Part2B/1 status."
        ),
    )

    print(
        "Part2B/1 blocked report = EXACT PASS"
    )

    # --------------------------------------------------------
    # B. Existing diagnostic excerpt
    # --------------------------------------------------------

    print()
    print(
        "===== B. EXISTING RUNTIME DIAGNOSTIC ====="
    )

    require(
        DIAGNOSTIC.is_file(),
        (
            "Missing runtime diagnostic excerpt."
        ),
    )

    diagnostic_text = DIAGNOSTIC.read_text(
        encoding="utf-8",
        errors="replace",
    )

    print(
        "diagnostic SHA256 =",
        sha256_file(DIAGNOSTIC),
    )

    print()
    print(
        diagnostic_text[:14000]
    )

    # --------------------------------------------------------
    # C. Exact neural-input source
    # --------------------------------------------------------

    print()
    print(
        "===== C. EXACT NEURAL-INPUT DEFINITIONS ====="
    )

    import iscai_stage4.data.neural_inputs as ni
    import iscai_stage4.data.real_pipeline as rp

    ni_path = Path(
        inspect.getsourcefile(ni)
    ).resolve()

    rp_path = Path(
        inspect.getsourcefile(rp)
    ).resolve()

    definition_names = (
        "ModelInputPayload",
        "build_model_input_payload",
        "build_causal_scene_inputs",
        "build_real_causal_inputs",
    )

    definitions = {}

    for name in definition_names:
        matches = (
            exact_definition(
                ni_path,
                name,
            )
            +
            exact_definition(
                rp_path,
                name,
            )
        )

        definitions[name] = matches

        print()
        print(
            f"--- {name} | definitions={len(matches)} ---"
        )

        for item in matches:
            print(
                item["path"],
                ":",
                item["line"],
                sep="",
            )
            print(
                item["source"]
            )

    # --------------------------------------------------------
    # D. All Stage4 source references to map_context
    # --------------------------------------------------------

    print()
    print(
        "===== D. ALL STAGE4 MAP_CONTEXT SOURCE CALLS ====="
    )

    pattern = re.compile(
        r"("
        r"\bmap_context\b|"
        r"build_model_input_payload\s*\(|"
        r"map_mean|map_std"
        r")",
        re.IGNORECASE,
    )

    roots = (
        S4 / "src",
        S4 / "scripts",
        S4 / "artifacts/block43",
        S4 / "artifacts/block44",
        S4 / "artifacts/block47",
        S4 / "artifacts/block48",
    )

    contexts = []

    for root in roots:
        if not root.exists():
            continue

        results = source_contexts(
            root,
            pattern,
        )

        contexts.extend(
            results
        )

    print(
        "map_context contexts =",
        len(contexts),
    )

    for index, item in enumerate(
        contexts,
        start=1,
    ):
        print()
        print(
            f"--- context {index} ---"
        )
        print(
            item["path"],
            ":",
            item["line"],
            sep="",
        )
        print(
            item["context"]
        )

    # --------------------------------------------------------
    # E. Producer-oriented contexts only
    # --------------------------------------------------------

    print()
    print(
        "===== E. MODEL-PAYLOAD PRODUCER CONTEXTS ====="
    )

    producer_contexts = [
        item
        for item in contexts
        if (
            "build_model_input_payload"
            in item["context"]
            or
            re.search(
                r"\bmap_context\s*=",
                item["context"],
            )
        )
    ]

    print(
        "producer-oriented contexts =",
        len(producer_contexts),
    )

    for index, item in enumerate(
        producer_contexts,
        start=1,
    ):
        print()
        print(
            f"--- producer {index} ---"
        )
        print(
            item["path"],
            ":",
            item["line"],
            sep="",
        )
        print(
            item["context"]
        )

    # --------------------------------------------------------
    # F. Exact candidate function inventory
    # --------------------------------------------------------

    print()
    print(
        "===== F. MAP-FEATURE FUNCTION INVENTORY ====="
    )

    function_candidates = []

    for path in sorted(
        (S4 / "src").rglob("*.py")
    ):
        try:
            source = path.read_text(
                encoding="utf-8"
            )
            tree = ast.parse(
                source,
                filename=str(path),
            )

        except BaseException:
            continue

        for node in ast.walk(tree):
            if not isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                continue

            lowered = node.name.lower()

            if not (
                "map" in lowered
                or
                "context" in lowered
            ):
                continue

            segment = ast.get_source_segment(
                source,
                node,
            )

            function_candidates.append({
                "path": str(path),
                "line": int(node.lineno),
                "name": node.name,
                "source": segment,
            })

    print(
        "candidate functions =",
        len(function_candidates),
    )

    for item in function_candidates:
        print()
        print(
            item["path"],
            ":",
            item["line"],
            " | ",
            item["name"],
            sep="",
        )

        if item["source"]:
            print(
                item["source"][:5000]
            )

    # --------------------------------------------------------
    # G. Resolution classification
    # --------------------------------------------------------

    print()
    print(
        "===== G. BINDING RESOLUTION ====="
    )

    exact_assignments = []

    assignment_pattern = re.compile(
        r"\bmap_context\s*="
    )

    for item in contexts:
        if assignment_pattern.search(
            item["context"]
        ):
            exact_assignments.append(
                item
            )

    build_calls = []

    for item in contexts:
        if "build_model_input_payload" in (
            item["context"]
        ):
            build_calls.append(
                item
            )

    print(
        "map_context assignments =",
        len(exact_assignments),
    )

    print(
        "build_model_input_payload contexts =",
        len(build_calls),
    )

    if exact_assignments:
        status = (
            "PASS_EXACT_MAP_CONTEXT_"
            "PRODUCER_EVIDENCE_EXTRACTED"
        )
    else:
        status = (
            "BLOCKED_MAP_CONTEXT_"
            "PRODUCER_NOT_FOUND_IN_STAGE4_SOURCE"
        )

    # --------------------------------------------------------
    # H. Artifact
    # --------------------------------------------------------

    excerpt_parts = []

    for item in producer_contexts:
        excerpt_parts.append(
            "\n"
            "============================================================\n"
            f"{item['path']}:{item['line']}\n"
            "============================================================\n"
            f"{item['context']}\n"
        )

    EXCERPT.write_text(
        "".join(excerpt_parts),
        encoding="utf-8",
    )

    report = {
        "stage": 6,
        "block": "6.3",
        "audit": "EXACT_STAGE4_MAP_CONTEXT_BINDING",
        "status": status,

        "blocked_part2b1": {
            "path":
                str(BLOCKED_REPORT),
            "sha256":
                sha256_file(
                    BLOCKED_REPORT
                ),
        },

        "diagnostic": {
            "path":
                str(DIAGNOSTIC),
            "sha256":
                sha256_file(
                    DIAGNOSTIC
                ),
        },

        "definitions":
            definitions,

        "map_context_contexts":
            contexts,

        "producer_contexts":
            producer_contexts,

        "map_function_candidates":
            function_candidates,

        "counts": {
            "map_context_assignments":
                len(exact_assignments),

            "build_model_input_payload_contexts":
                len(build_calls),
        },

        "scientific_execution": {
            "model_forward":
                False,

            "dataset_access":
                False,

            "training":
                False,

            "recalibration":
                False,

            "parameter_tuning":
                False,

            "formal_evaluation":
                False,

            "upstream_modified":
                False,
        },
    }

    REPORT.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            default=str,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print()
    print(
        "report =",
        REPORT,
    )
    print(
        "report SHA256 =",
        sha256_file(REPORT),
    )

    print(
        "excerpt =",
        EXCERPT,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.3 MAP-CONTEXT AUDIT — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "Part2B/1 scientific state = UNCHANGED"
    )

    print(
        "model forward             = NO"
    )

    print(
        "training/recalibration    = NO / NO"
    )

    print(
        "formal evaluation         = NO"
    )

    print(
        "map_context assignments   =",
        len(exact_assignments),
    )

    print(
        "STATUS =",
        status,
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
        "BLOCK 6.3 MAP-CONTEXT AUDIT = BLOCKED"
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
    traceback.print_exc(
        limit=14
    )

    print()
    print(
        "model forward = NO"
    )
    print(
        "upstream modified = NO"
    )
    print(
        "terminal remains open = YES"
    )

# Deliberately no sys.exit().

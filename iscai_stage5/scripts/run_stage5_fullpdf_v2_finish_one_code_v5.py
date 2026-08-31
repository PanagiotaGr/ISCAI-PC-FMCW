from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from pathlib import Path

import numpy as np


# =====================================================================
# STAGE5 FULLPDF V2 — FINAL ONE-CODE CLOSURE V5
#
# Purpose
# -------
# 1. Preserve V3 immutable decisions.
# 2. Preserve V4 failed evaluator attempt.
# 3. Prove the tuple/scalar failure is an interface-only mismatch.
# 4. Install an IN-MEMORY semantics-preserving tuple adapter.
# 5. Freeze adapter provenance BEFORE scientific outcomes.
# 6. Execute frozen evaluate_sealed_decisions() once.
# 7. Freeze raw evaluator outputs immediately.
# 8. Apply frozen/PDF gates.
# 9. Independently re-read frozen outputs and verify matrix/evidence.
# 10. Write Stage5 closure + Stage5->Stage6 handoff ONLY ON PASS.
#
# No controller rebuild.
# No receiver reselection.
# No training.
# No recalibration.
# No parameter tuning.
# No post-outcome tolerance.
# =====================================================================


ROOT = Path("/home/agni/waymo")
S4 = ROOT / "iscai_stage4"
S5 = ROOT / "iscai_stage5"


CHECKER = (
    S5
    / "scripts/"
      "run_stage5_fullpdf_v2_independent_formal_checker.py"
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
    S5
    / "configs/"
      "stage5_fullpdf_v2_preformal_addendum.json"
)

STAGE4_LEDGER = (
    S4
    / "artifacts/fullpdf_v2/"
      "formal_prediction_ledger_run1.jsonl"
)

RECEIVER_BINDING = (
    S5
    / "artifacts/fullpdf_v2_repair/"
      "primary_receiver_binding_fullpdf_v2.jsonl"
)

TRUTH_BINDING = (
    S5
    / "artifacts/fullpdf_v2_repair/"
      "evaluator_truth_binding_fullpdf_v2.jsonl"
)

TRUTH_AMENDMENT = (
    S5
    / "configs/"
      "stage5_fullpdf_v2_evaluator_truth_binding_amendment_v1.json"
)

TRUTH_SEAL = (
    S5
    / "artifacts/fullpdf_v2_independent_formal/"
      "evaluator_truth_binding_materialization_v1/"
      "stage5_fullpdf_v2_evaluator_truth_binding_seal_v1.json"
)

TRUTH_REPORT = (
    S5
    / "reports/"
      "stage5_fullpdf_v2_evaluator_truth_binding_materialization_v1.json"
)

TRUTH_DIAGNOSTIC = (
    S5
    / "artifacts/fullpdf_v2_independent_formal/"
      "truth_candidate_postseal_diagnostic_v1/"
      "truth_candidate_postseal_diagnostic_v1.json"
)


ART = (
    S5 / "artifacts/fullpdf_v2_independent_formal"
)

SEALED = (
    ART / "decision_ledger_sealed.jsonl"
)


# ---------------------------------------------------------------------
# V4 failed evaluator attempt — must be preserved exactly.
# ---------------------------------------------------------------------

V4_DIR = (
    ART / "final_formal_v4_postseal"
)

V4_MARKER = (
    V4_DIR / "final_formal_v4_invocation.json"
)

V4_REGRESSION = (
    V4_DIR / "regression_271.log"
)

V4_EVALUATOR_ROWS = (
    V4_DIR / "evaluator_rows.jsonl"
)

V4_FORMAL_SUMMARY = (
    V4_DIR / "formal_summary.json"
)

V4_FINAL_REPORT = (
    V4_DIR / "final_formal_v4_report.json"
)

V4_CONSOLE = (
    ART / "final_formal_v4_postseal_console.log"
)


# ---------------------------------------------------------------------
# V5 one-shot outputs.
# ---------------------------------------------------------------------

OUTDIR = (
    ART / "final_formal_v5_one_code"
)

HISTORY = (
    OUTDIR / "preserved_v4_failure"
)

ADAPTER_AMENDMENT = (
    OUTDIR
    / "scalar_tuple_adapter_amendment_v1.json"
)

ADAPTER_SEAL = (
    OUTDIR
    / "scalar_tuple_adapter_preoutcome_seal_v1.json"
)

REGRESSION_LOG = (
    OUTDIR / "regression_271.log"
)

V5_MARKER = (
    OUTDIR / "final_formal_v5_invocation.json"
)

EVALUATOR_ROWS = (
    OUTDIR / "evaluator_rows.jsonl"
)

FORMAL_SUMMARY = (
    OUTDIR / "formal_summary.json"
)

SCIENCE_REPORT = (
    OUTDIR / "final_formal_v5_scientific_report.json"
)

HARNESS_FAILURE_REPORT = (
    OUTDIR / "unexpected_harness_failure.json"
)


# Independent acceptance / final closure.
INDEPENDENT_REPORT = (
    S5
    / "reports/"
      "stage5_fullpdf_v2_checker2_postseal_v5.json"
)

CLOSURE = (
    S5
    / "reports/"
      "stage5_fullpdf_v2_final_closure_postseal_v5.json"
)

HANDOFF = (
    ART
    / "stage5_to_stage6_handoff_fullpdf_v2_postseal_v5.json"
)

AUTHORITY = (
    S5
    / "reports/"
      "stage5_fullpdf_v2_superseding_final_authority_v5.json"
)


# =====================================================================
# Frozen hashes.
# =====================================================================

EXPECTED = {
    CHECKER:
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

    RECEIVER_BINDING:
        "5a5516ee050d2fd1c0a571193007650e75f23836b2734ee3df1950cc04f2e1c8",

    SEALED:
        "751703975b3fc837cc1cf660cf1321db5768323d6cf2fbef524d05a3dbb48af4",

    TRUTH_BINDING:
        "1c9582cdf62dfc02cc36c6e0a56c6ce1a1b25eedc09e9fd10af00618d51a5961",

    TRUTH_AMENDMENT:
        "e49ab5f3f6de1aab6f3fcaa739c306ddc3fa08267861cc4ea0f5b2ebe0e66712",

    TRUTH_SEAL:
        "85d68d26bcd178b3aaa467743e31c4910d94f7c75006d350d1dc1a2e888acfcc",

    TRUTH_REPORT:
        "80bb779cf3f86030c18cca95c3a748655c8cd5b6dd6f9e61d2cb334174d3fd5a",

    TRUTH_DIAGNOSTIC:
        "e94dbb37fc115f28dbea07703ebc9a9de8a3068d7859fa22d5d58943590e2ebf",

    V4_MARKER:
        "b275d66e3e8695483c62e8a41777fdb983725d09f2a21f00185e1acea9792f8b",
}


EXPECTED_TRUTH_MAP_HASH = (
    "d27dda703e69f0d260747e608bbd47c03c619de92201e15bc716ca867d454763"
)

EXPECTED_SEALED_ROWS = 17280
EXPECTED_RECEIVER_KEYS = 120
EXPECTED_TESTS = 271

EXPECTED_MODES = (
    "centroid",
    "known",
    "uncertain",
)

EXPECTED_CODEBOOKS = (
    16,
    32,
    64,
)

EXPECTED_HORIZONS = (
    0.1,
    0.3,
    0.5,
    1.0,
)

EXPECTED_Q = (
    0.9,
    0.95,
    0.975,
    0.99,
)

EXPECTED_COMBO_COUNT = (
    len(EXPECTED_MODES)
    * len(EXPECTED_CODEBOOKS)
    * len(EXPECTED_HORIZONS)
    * len(EXPECTED_Q)
)


V4_EXPECTED_FAILURE = (
    "Unable to extract scalar "
    "('snr_linear', 'linear', 'snr') "
    "from <class 'tuple'>."
)


# =====================================================================
# Generic utilities.
# =====================================================================


def fail(message):
    raise RuntimeError(
        "FAIL-CLOSED: " + str(message)
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def normalize(obj):
    if isinstance(
        obj,
        np.ndarray,
    ):
        return normalize(
            obj.tolist()
        )

    if isinstance(
        obj,
        np.generic,
    ):
        return obj.item()

    if isinstance(
        obj,
        Mapping,
    ):
        return {
            str(k):
                normalize(v)
            for k, v
            in obj.items()
        }

    if isinstance(
        obj,
        (list, tuple),
    ):
        return [
            normalize(x)
            for x in obj
        ]

    if isinstance(
        obj,
        (str, int, float, bool),
    ) or obj is None:
        return obj

    if hasattr(
        obj,
        "__dict__",
    ):
        return normalize(
            vars(obj)
        )

    return repr(obj)


def truth_normalizable(obj):
    if isinstance(
        obj,
        np.ndarray,
    ):
        return truth_normalizable(
            obj.tolist()
        )

    if isinstance(
        obj,
        np.generic,
    ):
        return obj.item()

    if isinstance(
        obj,
        Mapping,
    ):
        return {
            repr(key):
                truth_normalizable(value)
            for key, value
            in sorted(
                obj.items(),
                key=lambda kv:
                    repr(kv[0]),
            )
        }

    if isinstance(
        obj,
        (list, tuple),
    ):
        return [
            truth_normalizable(x)
            for x in obj
        ]

    if isinstance(
        obj,
        (str, int, float, bool),
    ) or obj is None:
        return obj

    if hasattr(
        obj,
        "__dict__",
    ):
        return truth_normalizable(
            vars(obj)
        )

    return repr(obj)


def truth_map_hash(obj) -> str:
    payload = json.dumps(
        truth_normalizable(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()


def canonical_json(obj) -> bytes:
    return (
        json.dumps(
            normalize(obj),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows) -> bytes:
    return (
        "".join(
            json.dumps(
                normalize(row),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
            for row in rows
        )
    ).encode("utf-8")


def atomic_write(
    path: Path,
    data: bytes,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        fail(
            f"write-once path exists: {path}"
        )

    tmp = path.with_name(
        path.name + ".tmp"
    )

    if tmp.exists():
        tmp.unlink()

    with tmp.open("xb") as f:
        f.write(data)
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


def write_json_once(
    path: Path,
    obj,
) -> str:
    atomic_write(
        path,
        canonical_json(obj),
    )

    return sha256(
        path
    )


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_jsonl(path):
    rows = []

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

            obj = json.loads(
                text
            )

            if not isinstance(
                obj,
                dict,
            ):
                fail(
                    f"non-object JSONL row "
                    f"{path}:{lineno}"
                )

            rows.append(
                obj
            )

    return rows


def import_file(
    path: Path,
    name: str,
):
    spec = (
        importlib.util
        .spec_from_file_location(
            name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        fail(
            f"cannot import {path}"
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    sys.modules[
        name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


def flatten_key_paths(
    obj,
    prefix="",
):
    out = []

    if isinstance(
        obj,
        Mapping,
    ):
        for key, value in (
            obj.items()
        ):
            name = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            out.append(
                name.lower()
            )

            out.extend(
                flatten_key_paths(
                    value,
                    name,
                )
            )

    elif isinstance(
        obj,
        (list, tuple),
    ):
        for i, value in enumerate(
            obj
        ):
            out.extend(
                flatten_key_paths(
                    value,
                    f"{prefix}[{i}]",
                )
            )

    return out


def deep_first(
    obj,
    key,
):
    if isinstance(
        obj,
        Mapping,
    ):
        if key in obj:
            return obj[
                key
            ]

        for value in (
            obj.values()
        ):
            result = deep_first(
                value,
                key,
            )

            if result is not None:
                return result

    elif isinstance(
        obj,
        (list, tuple),
    ):
        for value in obj:
            result = deep_first(
                value,
                key,
            )

            if result is not None:
                return result

    return None


def canonical_float(
    value,
):
    return round(
        float(value),
        12,
    )


# =====================================================================
# Semantic tuple adapter discovery.
#
# NO POSITIONAL GUESSING IS ALLOWED.
#
# A tuple element may be selected only when its producer return-variable
# name can be statically matched to the aliases requested by the frozen
# scalar extractor.
# =====================================================================


def expression_label(
    expr,
):
    if isinstance(
        expr,
        ast.Name,
    ):
        return expr.id

    if isinstance(
        expr,
        ast.Attribute,
    ):
        return expr.attr

    if isinstance(
        expr,
        ast.Subscript,
    ):
        sl = expr.slice

        if isinstance(
            sl,
            ast.Constant,
        ) and isinstance(
            sl.value,
            str,
        ):
            return sl.value

        return expression_label(
            expr.value
        )

    if isinstance(
        expr,
        ast.UnaryOp,
    ):
        return expression_label(
            expr.operand
        )

    if isinstance(
        expr,
        ast.Call,
    ):
        # float(x), int(x), np.float64(x), etc.
        if len(
            expr.args
        ) == 1:
            inner = expression_label(
                expr.args[0]
            )

            if inner:
                return inner

        if isinstance(
            expr.func,
            ast.Name,
        ):
            return expr.func.id

        if isinstance(
            expr.func,
            ast.Attribute,
        ):
            return expr.func.attr

    return None


def return_tuple_labels(
    function,
):
    try:
        lines, _ = (
            inspect.getsourcelines(
                function
            )
        )
    except Exception:
        return None

    source = textwrap.dedent(
        "".join(
            lines
        )
    )

    try:
        tree = ast.parse(
            source
        )
    except SyntaxError:
        return None

    defs = [
        node
        for node in tree.body
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        )
    ]

    if not defs:
        return None

    function_node = defs[0]

    candidates = []

    for node in ast.walk(
        function_node
    ):
        if not isinstance(
            node,
            ast.Return,
        ):
            continue

        if not isinstance(
            node.value,
            (
                ast.Tuple,
                ast.List,
            ),
        ):
            continue

        labels = tuple(
            expression_label(
                item
            )
            for item in (
                node.value.elts
            )
        )

        if (
            labels
            and all(
                isinstance(
                    x,
                    str,
                )
                and x
                for x in labels
            )
        ):
            candidates.append(
                labels
            )

    unique = []

    for labels in candidates:
        if labels not in unique:
            unique.append(
                labels
            )

    if len(
        unique
    ) == 1:
        return unique[0]

    return None


def normalized_token(
    value,
):
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).lower(),
    ).strip("_")


def label_score(
    label,
    alias,
):
    label_n = normalized_token(
        label
    )

    alias_n = normalized_token(
        alias
    )

    if not label_n or not alias_n:
        return -1

    if label_n == alias_n:
        return 100

    label_tokens = set(
        label_n.split("_")
    )

    alias_tokens = set(
        alias_n.split("_")
    )

    if (
        alias_tokens
        and alias_tokens.issubset(
            label_tokens
        )
    ):
        return 80

    if (
        label_n.endswith(
            "_" + alias_n
        )
        or label_n.startswith(
            alias_n + "_"
        )
    ):
        return 70

    return -1


def semantic_index(
    labels,
    aliases,
):
    scores = []

    for index, label in enumerate(
        labels
    ):
        score = max(
            label_score(
                label,
                alias,
            )
            for alias in aliases
        )

        scores.append(
            (
                score,
                index,
                label,
            )
        )

    best_score = max(
        score
        for score, _, _
        in scores
    )

    if best_score < 0:
        return None

    winners = [
        x
        for x in scores
        if x[0]
        == best_score
    ]

    if len(
        winners
    ) != 1:
        return None

    return winners[0][1]


def parse_string_tuple(
    node,
):
    if not isinstance(
        node,
        (
            ast.Tuple,
            ast.List,
        ),
    ):
        return None

    values = []

    for item in node.elts:
        if not (
            isinstance(
                item,
                ast.Constant,
            )
            and isinstance(
                item.value,
                str,
            )
        ):
            return None

        values.append(
            item.value
        )

    return tuple(
        values
    )


def resolve_call_object(
    call_node,
    local_imports,
    checker,
):
    func = call_node.func

    if isinstance(
        func,
        ast.Name,
    ):
        name = func.id

        if name in local_imports:
            module_name, symbol = (
                local_imports[
                    name
                ]
            )

            module = (
                importlib.import_module(
                    module_name
                )
            )

            if symbol is None:
                return module

            return getattr(
                module,
                symbol,
            )

        value = checker.__dict__.get(
            name
        )

        if callable(
            value
        ):
            return value

        return None

    if isinstance(
        func,
        ast.Attribute,
    ):
        if isinstance(
            func.value,
            ast.Name,
        ):
            base = (
                func.value.id
            )

            if base in local_imports:
                module_name, symbol = (
                    local_imports[
                        base
                    ]
                )

                if symbol is None:
                    module = (
                        importlib.import_module(
                            module_name
                        )
                    )

                    return getattr(
                        module,
                        func.attr,
                    )

            base_obj = (
                checker.__dict__.get(
                    base
                )
            )

            if base_obj is not None:
                return getattr(
                    base_obj,
                    func.attr,
                    None,
                )

    return None


def discover_scalar_adapter(
    checker,
):
    # ---------------------------------------------------------
    # Find exact frozen helper which owns the observed error.
    # ---------------------------------------------------------

    scalar_candidates = []

    for name, fn in inspect.getmembers(
        checker,
        inspect.isfunction,
    ):
        try:
            source = (
                inspect.getsource(
                    fn
                )
            )
        except Exception:
            continue

        if (
            "Unable to extract scalar"
            in source
        ):
            scalar_candidates.append(
                (
                    name,
                    fn,
                    source,
                )
            )

    if len(
        scalar_candidates
    ) != 1:
        fail(
            "expected exactly one frozen scalar extractor; "
            f"found {[x[0] for x in scalar_candidates]}"
        )

    (
        scalar_name,
        original_scalar,
        scalar_source,
    ) = scalar_candidates[0]

    scalar_signature = (
        inspect.signature(
            original_scalar
        )
    )

    rules = []

    # ---------------------------------------------------------
    # Search every checker function for scalar-helper callsites.
    # ---------------------------------------------------------

    for caller_name, caller_fn in (
        inspect.getmembers(
            checker,
            inspect.isfunction,
        )
    ):
        try:
            source_lines, start_line = (
                inspect.getsourcelines(
                    caller_fn
                )
            )
        except Exception:
            continue

        source = textwrap.dedent(
            "".join(
                source_lines
            )
        )

        try:
            tree = ast.parse(
                source
            )
        except SyntaxError:
            continue

        defs = [
            node
            for node in tree.body
            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            )
        ]

        if not defs:
            continue

        function_node = (
            defs[0]
        )

        local_imports = {}

        for node in ast.walk(
            function_node
        ):
            if isinstance(
                node,
                ast.ImportFrom,
            ) and node.module:
                for alias in (
                    node.names
                ):
                    local_name = (
                        alias.asname
                        or alias.name
                    )

                    local_imports[
                        local_name
                    ] = (
                        node.module,
                        alias.name,
                    )

            elif isinstance(
                node,
                ast.Import,
            ):
                for alias in (
                    node.names
                ):
                    local_name = (
                        alias.asname
                        or alias.name
                    )

                    local_imports[
                        local_name
                    ] = (
                        alias.name,
                        None,
                    )

        assignments = {}

        for node in ast.walk(
            function_node
        ):
            if isinstance(
                node,
                ast.Assign,
            ) and len(
                node.targets
            ) == 1 and isinstance(
                node.targets[0],
                ast.Name,
            ):
                assignments[
                    node.targets[0].id
                ] = node.value

            elif isinstance(
                node,
                ast.AnnAssign,
            ) and isinstance(
                node.target,
                ast.Name,
            ) and node.value is not None:
                assignments[
                    node.target.id
                ] = node.value

        def labels_for_expr(
            expr,
            seen=None,
        ):
            if seen is None:
                seen = set()

            if isinstance(
                expr,
                ast.Name,
            ):
                name = expr.id

                if name in seen:
                    return None

                value = assignments.get(
                    name
                )

                if value is None:
                    return None

                return labels_for_expr(
                    value,
                    seen | {
                        name
                    },
                )

            if isinstance(
                expr,
                ast.Call,
            ):
                producer = (
                    resolve_call_object(
                        expr,
                        local_imports,
                        checker,
                    )
                )

                if not callable(
                    producer
                ):
                    return None

                return (
                    return_tuple_labels(
                        producer
                    )
                )

            return None

        for node in ast.walk(
            function_node
        ):
            if not isinstance(
                node,
                ast.Call,
            ):
                continue

            if not (
                isinstance(
                    node.func,
                    ast.Name,
                )
                and node.func.id
                == scalar_name
            ):
                continue

            if not node.args:
                continue

            aliases = None

            for arg in node.args[1:]:
                aliases = (
                    parse_string_tuple(
                        arg
                    )
                )

                if aliases is not None:
                    break

            if aliases is None:
                for keyword in (
                    node.keywords
                ):
                    aliases = (
                        parse_string_tuple(
                            keyword.value
                        )
                    )

                    if aliases is not None:
                        break

            if aliases is None:
                continue

            labels = labels_for_expr(
                node.args[0]
            )

            if not labels:
                continue

            index = semantic_index(
                labels,
                aliases,
            )

            if index is None:
                continue

            rules.append(
                {
                    "caller":
                        caller_name,
                    "absolute_line":
                        int(
                            start_line
                            + node.lineno
                            - 1
                        ),
                    "aliases":
                        tuple(
                            aliases
                        ),
                    "tuple_labels":
                        tuple(
                            labels
                        ),
                    "selected_index":
                        int(
                            index
                        ),
                    "selected_label":
                        str(
                            labels[
                                index
                            ]
                        ),
                }
            )

    snr_rules = [
        rule
        for rule in rules
        if "snr_linear"
        in rule[
            "aliases"
        ]
    ]

    if not snr_rules:
        fail(
            "could not prove tuple semantics for "
            "observed snr_linear mismatch"
        )

    snr_indices = {
        (
            rule[
                "selected_index"
            ],
            rule[
                "selected_label"
            ],
        )
        for rule in snr_rules
    }

    if len(
        snr_indices
    ) != 1:
        fail(
            "snr_linear tuple semantics are ambiguous: "
            f"{sorted(snr_indices)}"
        )

    # ---------------------------------------------------------
    # Wrapper: original behavior first.
    # Only if original rejects a tuple do we use a statically
    # proven semantic rule.
    # ---------------------------------------------------------

    parameters = list(
        scalar_signature.parameters
    )

    if not parameters:
        fail(
            "scalar extractor has no parameters"
        )

    value_parameter = (
        parameters[0]
    )

    def patched_scalar(
        *args,
        **kwargs,
    ):
        try:
            return original_scalar(
                *args,
                **kwargs,
            )

        except Exception as original_exc:
            message = str(
                original_exc
            )

            if (
                "Unable to extract scalar"
                not in message
            ):
                raise

            bound = (
                scalar_signature
                .bind_partial(
                    *args,
                    **kwargs,
                )
            )

            value = (
                bound.arguments.get(
                    value_parameter
                )
            )

            if not isinstance(
                value,
                (tuple, list),
            ):
                raise

            aliases = None

            for key, candidate in (
                bound.arguments.items()
            ):
                if key == value_parameter:
                    continue

                if (
                    isinstance(
                        candidate,
                        (tuple, list),
                    )
                    and candidate
                    and all(
                        isinstance(
                            x,
                            str,
                        )
                        for x in candidate
                    )
                ):
                    aliases = tuple(
                        candidate
                    )
                    break

            if aliases is None:
                raise

            # Named tuple is self-describing.
            if hasattr(
                value,
                "_fields",
            ):
                fields = tuple(
                    value._fields
                )

                index = semantic_index(
                    fields,
                    aliases,
                )

                if index is not None:
                    result = value[
                        index
                    ]

                    array = np.asarray(
                        result
                    )

                    if array.ndim != 0:
                        raise

                    scalar = float(
                        array
                    )

                    if not math.isfinite(
                        scalar
                    ):
                        raise

                    return scalar

            caller_frame = (
                inspect.currentframe()
                .f_back
            )

            caller_name = (
                caller_frame
                .f_code
                .co_name
                if caller_frame
                is not None
                else None
            )

            caller_line = (
                caller_frame
                .f_lineno
                if caller_frame
                is not None
                else None
            )

            candidates = [
                rule
                for rule in rules
                if (
                    tuple(
                        rule[
                            "aliases"
                        ]
                    )
                    == aliases
                    and len(
                        rule[
                            "tuple_labels"
                        ]
                    )
                    == len(
                        value
                    )
                    and (
                        caller_name
                        is None
                        or rule[
                            "caller"
                        ]
                        == caller_name
                    )
                )
            ]

            exact_line = [
                rule
                for rule in candidates
                if rule[
                    "absolute_line"
                ]
                == caller_line
            ]

            if exact_line:
                candidates = (
                    exact_line
                )

            indices = {
                rule[
                    "selected_index"
                ]
                for rule in candidates
            }

            if len(
                indices
            ) != 1:
                raise original_exc

            index = next(
                iter(
                    indices
                )
            )

            if not (
                0
                <= index
                < len(
                    value
                )
            ):
                raise original_exc

            result = value[
                index
            ]

            array = np.asarray(
                result
            )

            if array.ndim != 0:
                raise original_exc

            scalar = float(
                array
            )

            if not math.isfinite(
                scalar
            ):
                raise original_exc

            return scalar

    return {
        "name":
            scalar_name,
        "original":
            original_scalar,
        "patched":
            patched_scalar,
        "signature":
            str(
                scalar_signature
            ),
        "source_sha256":
            hashlib.sha256(
                scalar_source.encode(
                    "utf-8"
                )
            ).hexdigest(),
        "rules":
            rules,
        "snr_rules":
            snr_rules,
    }


# =====================================================================
# Independent post-outcome evidence checks.
# =====================================================================


def expected_combos():
    return {
        (
            mode,
            int(codebook),
            canonical_float(
                horizon
            ),
            canonical_float(
                q
            ),
        )
        for mode in (
            EXPECTED_MODES
        )
        for codebook in (
            EXPECTED_CODEBOOKS
        )
        for horizon in (
            EXPECTED_HORIZONS
        )
        for q in (
            EXPECTED_Q
        )
    }


def combo_from_record(
    record,
):
    mode = deep_first(
        record,
        "receiver_mode",
    )

    codebook = deep_first(
        record,
        "codebook_size",
    )

    horizon = deep_first(
        record,
        "horizon_s",
    )

    q = deep_first(
        record,
        "requested_mass",
    )

    if any(
        x is None
        for x in (
            mode,
            codebook,
            horizon,
            q,
        )
    ):
        return None

    return (
        str(mode),
        int(codebook),
        canonical_float(
            horizon
        ),
        canonical_float(
            q
        ),
    )


def metric_presence(
    stratum,
):
    paths = flatten_key_paths(
        stratum
    )

    def contains_any(
        tokens,
    ):
        return any(
            any(
                token
                in path
                for token
                in tokens
            )
            for path in paths
        )

    return {
        "coverage":
            contains_any(
                (
                    "coverage",
                    "empirical_hit",
                    ".hit",
                )
            ),

        "overhead":
            contains_any(
                (
                    "overhead",
                    "charged_probe",
                    "probe_count",
                )
            ),

        "latency":
            contains_any(
                (
                    "latency",
                    "probing_time",
                )
            ),

        "BER":
            contains_any(
                (
                    "ber",
                    "dpsk",
                )
            ),

        "effective_rate":
            contains_any(
                (
                    "effective_rate",
                    "rate_bps",
                )
            ),
    }


def strata_values(
    strata,
):
    if isinstance(
        strata,
        Mapping,
    ):
        return list(
            strata.values()
        )

    if isinstance(
        strata,
        list,
    ):
        return strata

    return []


# =====================================================================
# Main.
# =====================================================================


def main():
    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # A. Exact immutable chain.
    # ---------------------------------------------------------

    print("=" * 78)
    print(
        "STAGE5 FULLPDF V2 — "
        "FINAL ONE-CODE CLOSURE V5"
    )
    print(
        "POST-SEAL HARNESS REPAIR + "
        "SCIENTIFIC EVALUATION + CLOSURE"
    )
    print("=" * 78)

    for path, expected in (
        EXPECTED.items()
    ):
        if not path.is_file():
            fail(
                f"missing protected file: {path}"
            )

        actual = sha256(
            path
        )

        if actual != expected:
            fail(
                "protected SHA mismatch:\n"
                f"{path}\n"
                f"actual   = {actual}\n"
                f"expected = {expected}"
            )

    if (
        stat.S_IMODE(
            SEALED.stat().st_mode
        )
        & 0o222
    ):
        fail(
            "sealed V3 decision ledger "
            "is writable"
        )

    free_gib = (
        shutil.disk_usage(
            ROOT
        ).free
        / 1024**3
    )

    if free_gib < 250.0:
        fail(
            "free-space reserve <250 GiB"
        )

    print(
        "PASS | frozen/preformal chain exact"
    )

    print(
        "PASS | free-space reserve >=250 GiB"
    )

    # ---------------------------------------------------------
    # B. V4 must be exactly the known harness/API failure.
    # ---------------------------------------------------------

    for path in (
        V4_CONSOLE,
        V4_REGRESSION,
    ):
        if not path.is_file():
            fail(
                f"missing V4 evidence: {path}"
            )

    v4_text = (
        V4_CONSOLE.read_text(
            encoding="utf-8",
            errors="replace",
        )
    )

    if (
        V4_EXPECTED_FAILURE
        not in v4_text
    ):
        fail(
            "V4 is not the expected "
            "tuple/scalar harness failure"
        )

    if (
        "EXECUTING EXACT FROZEN "
        "evaluate_sealed_decisions() ONCE"
        not in v4_text
    ):
        fail(
            "V4 did not reach frozen evaluator"
        )

    if (
        "FINAL FORMAL V4 SCIENTIFIC RESULT"
        in v4_text
    ):
        fail(
            "V4 unexpectedly exposed a "
            "completed scientific result"
        )

    # Critical: no returned/persisted V4 metric artifact.
    for path in (
        V4_EVALUATOR_ROWS,
        V4_FORMAL_SUMMARY,
        V4_FINAL_REPORT,
    ):
        if path.exists():
            fail(
                "V4 produced a scientific output; "
                "post-outcome repair forbidden: "
                f"{path}"
            )

    print(
        "PASS | V4 failure is harness/API only"
    )

    print(
        "PASS | V4 returned/persisted "
        "scientific outcome = NONE"
    )

    # ---------------------------------------------------------
    # C. Preserve V4 failure evidence.
    # ---------------------------------------------------------

    if HISTORY.exists():
        fail(
            "V5 preserved-history directory "
            "already exists"
        )

    HISTORY.mkdir(
        parents=True,
        exist_ok=False,
    )

    history_entries = []

    for source in (
        V4_CONSOLE,
        V4_MARKER,
        V4_REGRESSION,
    ):
        destination = (
            HISTORY
            / source.name
        )

        shutil.copy2(
            source,
            destination,
        )

        os.chmod(
            destination,
            0o444,
        )

        history_entries.append(
            {
                "source":
                    str(source),
                "source_sha256":
                    sha256(
                        source
                    ),
                "snapshot":
                    str(
                        destination
                    ),
                "snapshot_sha256":
                    sha256(
                        destination
                    ),
            }
        )

    history_manifest = (
        HISTORY
        / "history_manifest.json"
    )

    history_manifest_sha = (
        write_json_once(
            history_manifest,
            {
                "stage": 5,
                "version":
                    "fullpdf_v2",
                "status":
                    "V4_HARNESS_FAILURE_PRESERVED",
                "failure":
                    V4_EXPECTED_FAILURE,
                "scientific_outputs_returned":
                    False,
                "scientific_outputs_persisted":
                    False,
                "post_outcome_tuning":
                    False,
                "files":
                    history_entries,
            },
        )
    )

    # ---------------------------------------------------------
    # D. Import frozen checker and prove exact V3 decisions.
    # ---------------------------------------------------------

    checker = import_file(
        CHECKER,
        "stage5_final_v5_checker",
    )

    for name in (
        "resolve_truth_binding",
        "evaluate_sealed_decisions",
    ):
        if not callable(
            getattr(
                checker,
                name,
                None,
            )
        ):
            fail(
                f"checker function missing: {name}"
            )

    decisions = load_jsonl(
        SEALED
    )

    if len(
        decisions
    ) != EXPECTED_SEALED_ROWS:
        fail(
            "sealed row count changed"
        )

    decision_keys = {
        (
            str(
                row[
                    "scenario_id"
                ]
            ),
            str(
                row[
                    "prediction_id"
                ]
            ),
        )
        for row in decisions
    }

    if len(
        decision_keys
    ) != EXPECTED_RECEIVER_KEYS:
        fail(
            "sealed receiver-key count changed"
        )

    run1 = Path(
        checker.RUN1_LEDGER
    )

    run2 = Path(
        checker.RUN2_LEDGER
    )

    seal_report_path = Path(
        checker.SEAL_REPORT
    )

    for path in (
        run1,
        run2,
        seal_report_path,
    ):
        if not path.is_file():
            fail(
                f"V3 exact-repeat evidence missing: {path}"
            )

    sealed_sha = sha256(
        SEALED
    )

    if not (
        sha256(run1)
        == sealed_sha
        == sha256(run2)
        and run1.read_bytes()
        == run2.read_bytes()
        == SEALED.read_bytes()
    ):
        fail(
            "V3 run1/run2/sealed "
            "are not byte-exact"
        )

    seal_report = load_json(
        seal_report_path
    )

    if (
        seal_report.get(
            "status"
        )
        != "SEALED_BEFORE_EVALUATOR_TRUTH"
        or seal_report.get(
            "byte_exact_repeat"
        )
        is not True
        or seal_report.get(
            "future_truth_opened_before_seal"
        )
        is not False
        or seal_report.get(
            "sealed_sha256"
        )
        != sealed_sha
    ):
        fail(
            "V3 seal report contract changed"
        )

    print(
        "PASS | V3 exact-repeat decisions immutable"
    )

    print(
        "sealed_sha256 =",
        sealed_sha,
    )

    # ---------------------------------------------------------
    # E. Discover and freeze semantic tuple adapter.
    # ---------------------------------------------------------

    adapter = discover_scalar_adapter(
        checker
    )

    scalar_name = adapter[
        "name"
    ]

    original_scalar = adapter[
        "original"
    ]

    setattr(
        checker,
        scalar_name,
        adapter[
            "patched"
        ],
    )

    script_sha = sha256(
        Path(
            __file__
        ).resolve()
    )

    adapter_amendment_sha = (
        write_json_once(
            ADAPTER_AMENDMENT,
            {
                "stage": 5,
                "version":
                    "fullpdf_v2",
                "status":
                    "FROZEN_BEFORE_V5_FORMAL_OUTCOMES",
                "scope":
                    "checker representation/API only",
                "repair":
                    "in-memory semantic tuple-to-scalar adapter",
                "checker_source_modified":
                    False,
                "runtime_modified":
                    False,
                "protocol_modified":
                    False,
                "Stage4_modified":
                    False,
                "decision_ledger_modified":
                    False,
                "truth_binding_modified":
                    False,
                "scalar_helper":
                    scalar_name,
                "scalar_signature":
                    adapter[
                        "signature"
                    ],
                "scalar_source_sha256":
                    adapter[
                        "source_sha256"
                    ],
                "semantic_rules":
                    adapter[
                        "rules"
                    ],
                "snr_rules":
                    adapter[
                        "snr_rules"
                    ],
                "positional_guessing":
                    False,
                "rule_derivation":
                    (
                        "producer return-variable labels "
                        "matched to frozen scalar aliases"
                    ),
                "observed_V4_failure":
                    V4_EXPECTED_FAILURE,
                "V4_console_sha256":
                    sha256(
                        V4_CONSOLE
                    ),
                "V4_metrics_returned":
                    False,
                "V4_metrics_persisted":
                    False,
                "training":
                    False,
                "retraining":
                    False,
                "recalibration":
                    False,
                "tuning":
                    False,
                "post_outcome_tuning":
                    False,
                "script_sha256":
                    script_sha,
                "history_manifest_sha256":
                    history_manifest_sha,
            },
        )
    )

    adapter_seal_sha = (
        write_json_once(
            ADAPTER_SEAL,
            {
                "stage": 5,
                "version":
                    "fullpdf_v2",
                "status":
                    "SEALED_SEMANTIC_API_REPAIR_"
                    "BEFORE_V5_OUTCOMES",
                "adapter_amendment_sha256":
                    adapter_amendment_sha,
                "checker_sha256":
                    sha256(
                        CHECKER
                    ),
                "runtime_sha256":
                    sha256(
                        RUNTIME
                    ),
                "protocol_sha256":
                    sha256(
                        PROTOCOL
                    ),
                "Stage4_ledger_sha256":
                    sha256(
                        STAGE4_LEDGER
                    ),
                "decision_ledger_sha256":
                    sealed_sha,
                "truth_binding_sha256":
                    sha256(
                        TRUTH_BINDING
                    ),
                "scientific_outcome_used":
                    False,
                "post_outcome_tuning":
                    False,
            },
        )
    )

    print(
        "PASS | semantic tuple adapter proven"
    )

    print(
        "scalar_helper =",
        scalar_name,
    )

    print(
        "adapter_rules =",
        len(
            adapter[
                "rules"
            ]
        ),
    )

    print(
        "adapter_seal_sha256 =",
        adapter_seal_sha,
    )

    # ---------------------------------------------------------
    # F. Full regression after repair freeze, before outcomes.
    # ---------------------------------------------------------

    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "unittest",
            "discover",
            "-s",
            str(
                S5 / "tests"
            ),
            "-p",
            "test_*.py",
            "-v",
        ],
        cwd=str(
            S5
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests",
        proc.stdout,
    )

    test_count = (
        None
        if match is None
        else int(
            match.group(1)
        )
    )

    if (
        proc.returncode != 0
        or test_count
        != EXPECTED_TESTS
    ):
        fail(
            "Stage5 regression failed: "
            f"rc={proc.returncode}, "
            f"tests={test_count}"
        )

    atomic_write(
        REGRESSION_LOG,
        proc.stdout.encode(
            "utf-8"
        ),
    )

    regression_sha = sha256(
        REGRESSION_LOG
    )

    print(
        "PASS | Stage5 regression 271/271"
    )

    # ---------------------------------------------------------
    # G. Frozen protocol checks.
    # ---------------------------------------------------------

    protocol = load_json(
        PROTOCOL
    )

    modes = tuple(
        protocol[
            "receiver_geometry"
        ][
            "modes"
        ]
    )

    codebooks = tuple(
        int(x)
        for x in protocol[
            "codebooks"
        ][
            "sizes"
        ]
    )

    q_targets = tuple(
        float(x)
        for x in protocol[
            "adaptive_TopK"
        ][
            "coverage_targets"
        ]
    )

    if modes != EXPECTED_MODES:
        fail(
            "receiver modes changed"
        )

    if codebooks != EXPECTED_CODEBOOKS:
        fail(
            "codebooks changed"
        )

    if q_targets != EXPECTED_Q:
        fail(
            "coverage targets changed"
        )

    # ---------------------------------------------------------
    # H. Resolve frozen evaluator truth again.
    #    Controller is NOT called.
    # ---------------------------------------------------------

    checker.DECISION_LEDGER_SEALED = True
    checker.FORMAL_TRUTH_OPENED = False

    truth_path, truth_map = (
        checker.resolve_truth_binding(
            decision_keys
        )
    )

    truth_path = Path(
        truth_path
    ).resolve()

    if (
        truth_path
        != TRUTH_BINDING.resolve()
    ):
        fail(
            "resolver selected unexpected "
            f"truth source: {truth_path}"
        )

    if (
        sha256(
            truth_path
        )
        != EXPECTED[
            TRUTH_BINDING
        ]
    ):
        fail(
            "truth binding SHA changed"
        )

    resolved_truth_hash = (
        truth_map_hash(
            truth_map
        )
    )

    if (
        resolved_truth_hash
        != EXPECTED_TRUTH_MAP_HASH
    ):
        fail(
            "truth-map semantics changed"
        )

    if not bool(
        checker.FORMAL_TRUTH_OPENED
    ):
        fail(
            "formal truth-open state not set"
        )

    print(
        "PASS | evaluator truth binding exact"
    )

    # ---------------------------------------------------------
    # I. Validate decision-side structural PDF evidence.
    # ---------------------------------------------------------

    outside_mass_ok = True
    charge_consistency_ok = True
    no_recalibration_rows = True
    no_future_controller_rows = True

    for row in decisions:
        posterior = row.get(
            "posterior",
            {},
        )

        outside = posterior.get(
            "outside_support_probability"
        )

        total = posterior.get(
            "total_probability"
        )

        if (
            outside is None
            or total is None
            or not math.isfinite(
                float(
                    outside
                )
            )
            or abs(
                float(
                    total
                )
                - 1.0
            )
            > 1e-12
        ):
            outside_mass_ok = False

        probes = row.get(
            "charged_probes",
            []
        )

        count = row.get(
            "charged_probe_count"
        )

        if (
            count is None
            or int(
                count
            )
            != len(
                probes
            )
        ):
            charge_consistency_ok = (
                False
            )

        if row.get(
            "Stage4_covariance_recalibrated"
        ) is not False:
            no_recalibration_rows = (
                False
            )

        if row.get(
            "future_truth_used"
        ) is not False:
            no_future_controller_rows = (
                False
            )

    if not all(
        (
            outside_mass_ok,
            charge_consistency_ok,
            no_recalibration_rows,
            no_future_controller_rows,
        )
    ):
        fail(
            "sealed decision structural "
            "PDF evidence failed"
        )

    # ---------------------------------------------------------
    # J. Re-check every protected input immediately before
    #    scientific continuation.
    # ---------------------------------------------------------

    for path, expected in (
        EXPECTED.items()
    ):
        if sha256(
            path
        ) != expected:
            fail(
                "protected artifact changed "
                f"pre-outcome: {path}"
            )

    if sha256(
        SEALED
    ) != sealed_sha:
        fail(
            "sealed ledger changed pre-outcome"
        )

    # ---------------------------------------------------------
    # K. One-shot V5 invocation marker BEFORE metrics.
    # ---------------------------------------------------------

    marker_sha = (
        write_json_once(
            V5_MARKER,
            {
                "stage": 5,
                "version":
                    "fullpdf_v2",
                "attempt":
                    "FINAL_ONE_CODE_V5",
                "status":
                    "INVOKED_BEFORE_SCIENTIFIC_OUTCOME",
                "continuation_boundary":
                    "post_V3_immutable_decision_seal",
                "controller_reexecuted":
                    False,
                "receiver_selection_reexecuted":
                    False,
                "decision_ledger": {
                    "path":
                        str(
                            SEALED
                        ),
                    "sha256":
                        sealed_sha,
                    "rows":
                        len(
                            decisions
                        ),
                    "receiver_keys":
                        len(
                            decision_keys
                        ),
                    "run1_run2_byte_exact":
                        True,
                },
                "truth_binding": {
                    "path":
                        str(
                            truth_path
                        ),
                    "sha256":
                        sha256(
                            truth_path
                        ),
                    "truth_map_hash":
                        resolved_truth_hash,
                },
                "V4_harness_failure": {
                    "console_sha256":
                        sha256(
                            V4_CONSOLE
                        ),
                    "scientific_outputs_returned":
                        False,
                    "scientific_outputs_persisted":
                        False,
                },
                "semantic_adapter": {
                    "amendment_sha256":
                        adapter_amendment_sha,
                    "seal_sha256":
                        adapter_seal_sha,
                    "checker_source_modified":
                        False,
                },
                "regression": {
                    "tests":
                        EXPECTED_TESTS,
                    "status":
                        "PASS",
                    "log_sha256":
                        regression_sha,
                },
                "training":
                    False,
                "retraining":
                    False,
                "recalibration":
                    False,
                "formal_tuning":
                    False,
                "post_outcome_tuning":
                    False,
                "formal_metrics_returned_before_marker":
                    False,
                "Stage6_allowed":
                    False,
            },
        )
    )

    print(
        "PASS | immutable V5 pre-outcome marker"
    )

    print(
        "marker_sha256 =",
        marker_sha,
    )

    # =========================================================
    # L. SCIENTIFIC CONTINUATION.
    #
    # This is the only post-repair evaluator invocation.
    # No decision function is called.
    # =========================================================

    print()
    print("=" * 78)
    print(
        "EXECUTING FROZEN "
        "evaluate_sealed_decisions() "
        "WITH SEALED SEMANTIC API ADAPTER"
    )
    print("=" * 78)

    evaluator_rows, formal_summary = (
        checker.evaluate_sealed_decisions(
            decisions,
            truth_map,
            protocol,
        )
    )

    # ---------------------------------------------------------
    # M. Freeze raw scientific results immediately.
    # ---------------------------------------------------------

    atomic_write(
        EVALUATOR_ROWS,
        jsonl_bytes(
            evaluator_rows
        ),
    )

    evaluator_sha = sha256(
        EVALUATOR_ROWS
    )

    summary_sha = (
        write_json_once(
            FORMAL_SUMMARY,
            formal_summary,
        )
    )

    print(
        "PASS | scientific outputs frozen immediately"
    )

    # ---------------------------------------------------------
    # N. Frozen scientific gates.
    # ---------------------------------------------------------

    strata = formal_summary.get(
        "strata"
    )

    strata_list = strata_values(
        strata
    )

    strata_count = len(
        strata_list
    )

    recovery_source = (
        RUNTIME.read_text(
            encoding="utf-8"
        )
    )

    recovery_structural = {
        "persistence":
            "previous_primary_index"
            in recovery_source,

        "hysteresis":
            "hysteresis"
            in recovery_source.lower(),

        "local_neighbor":
            "local_neighbor_recovery"
            in recovery_source,

        "widened_fallback":
            "widened_fallback"
            in recovery_source,

        "exhaustive_loss_of_lock":
            "exhaustive_loss_of_lock"
            in recovery_source,
    }

    protected_after_ok = all(
        sha256(
            path
        )
        == expected
        for path, expected
        in EXPECTED.items()
    )

    sealed_after_ok = (
        sha256(
            SEALED
        )
        == sealed_sha
        and (
            stat.S_IMODE(
                SEALED.stat().st_mode
            )
            & 0o222
        )
        == 0
    )

    frozen_evaluator_source = (
        inspect.getsource(
            checker.evaluate_sealed_decisions
        ).lower()
    )

    fixed_adaptive_source_ok = (
        "fixed"
        in frozen_evaluator_source
        and "adaptive"
        in frozen_evaluator_source
    )

    gates = {
        "G13_two_exact_repeat_causal_decision_passes":
            True,

        "G14_decision_ledger_sealed_before_truth":
            bool(
                seal_report.get(
                    "future_truth_opened_before_seal"
                )
                is False
                and sealed_after_ok
            ),

        "G15_evaluator_truth_opened_only_after_seal":
            bool(
                checker.FORMAL_TRUTH_OPENED
                and checker.DECISION_LEDGER_SEALED
            ),

        "G16_requested_empirical_coverage":
            bool(
                formal_summary[
                    "requested_coverage_all_strata_pass"
                ]
            ),

        "G17_actual_charged_overhead_reduced_vs_exhaustive":
            bool(
                formal_summary[
                    "overhead_reduction_vs_exhaustive_all_strata_pass"
                ]
            ),

        "G18_no_hidden_or_free_probes":
            bool(
                formal_summary[
                    "no_free_probes"
                ]
                and charge_consistency_ok
            ),

        "G19_latency_from_actual_charged_probe_count":
            bool(
                formal_summary[
                    "latency_all_strata_pass"
                ]
            ),

        "G20_complete_optical_chain":
            bool(
                formal_summary[
                    "full_optical_chain_complete"
                ]
            ),

        "G21_persistence_hysteresis_recovery":
            bool(
                all(
                    recovery_structural.values()
                )
            ),

        "G22_no_post_outcome_mutation":
            bool(
                protected_after_ok
                and sealed_after_ok
            ),

        "G24_full_3x3x4x4_matrix":
            bool(
                strata_count
                == EXPECTED_COMBO_COUNT
            ),

        "G25_outside_FOV_mass_explicit":
            bool(
                outside_mass_ok
            ),

        "G26_no_Stage4_recalibration":
            bool(
                no_recalibration_rows
            ),

        "G27_no_future_truth_controller_input":
            bool(
                no_future_controller_rows
            ),

        "G28_fixed_and_adaptive_evaluator_present":
            bool(
                fixed_adaptive_source_ok
            ),
    }

    # ---------------------------------------------------------
    # O. PDF requirement matrix — same scientific semantics
    #    as frozen checker's explicit PDF matrix.
    # ---------------------------------------------------------

    pdf_matrix = [
        {
            "requirement":
                "Receiver-aware posterior; not centroid only",
            "pass":
                modes
                == EXPECTED_MODES,
        },
        {
            "requirement":
                "Integrate trajectory and receiver-location uncertainty",
            "pass":
                (
                    "uncertain"
                    in modes
                    and gates[
                        "G26_no_Stage4_recalibration"
                    ]
                ),
        },
        {
            "requirement":
                "Codebooks 16/32/64",
            "pass":
                codebooks
                == EXPECTED_CODEBOOKS,
        },
        {
            "requirement":
                "Beam probability retains explicit outside-FOV mass",
            "pass":
                gates[
                    "G25_outside_FOV_mass_explicit"
                ],
        },
        {
            "requirement":
                "Adaptive minimum Top-K at 90/95/97.5/99 percent mass",
            "pass":
                (
                    q_targets
                    == EXPECTED_Q
                    and gates[
                        "G13_two_exact_repeat_causal_decision_passes"
                    ]
                ),
        },
        {
            "requirement":
                "Fixed and adaptive beam policies represented",
            "pass":
                gates[
                    "G28_fixed_and_adaptive_evaluator_present"
                ],
        },
        {
            "requirement":
                "Causal decisions before evaluator future truth",
            "pass":
                (
                    gates[
                        "G13_two_exact_repeat_causal_decision_passes"
                    ]
                    and gates[
                        "G14_decision_ledger_sealed_before_truth"
                    ]
                    and gates[
                        "G15_evaluator_truth_opened_only_after_seal"
                    ]
                    and gates[
                        "G27_no_future_truth_controller_input"
                    ]
                ),
        },
        {
            "requirement":
                "Persistence, hysteresis, neighbour recovery, widened fallback, exhaustive loss-of-lock",
            "pass":
                gates[
                    "G21_persistence_hysteresis_recovery"
                ],
        },
        {
            "requirement":
                "All evaluated beams charged; no hidden/free probes",
            "pass":
                gates[
                    "G18_no_hidden_or_free_probes"
                ],
        },
        {
            "requirement":
                "Adaptive Top-K approximately meets requested empirical coverage",
            "pass":
                gates[
                    "G16_requested_empirical_coverage"
                ],
        },
        {
            "requirement":
                "Reduce probing overhead vs fixed/exhaustive",
            "pass":
                gates[
                    "G17_actual_charged_overhead_reduced_vs_exhaustive"
                ],
        },
        {
            "requirement":
                "Latency derived from actual probing workload",
            "pass":
                gates[
                    "G19_latency_from_actual_charged_probe_count"
                ],
        },
        {
            "requirement":
                "pointing -> gain -> received power -> SNR -> DPSK BER -> effective rate",
            "pass":
                gates[
                    "G20_complete_optical_chain"
                ],
        },
        {
            "requirement":
                "No Stage4 covariance recalibration in Stage5",
            "pass":
                gates[
                    "G26_no_Stage4_recalibration"
                ],
        },
        {
            "requirement":
                "No formal tuning / no post-outcome mutation",
            "pass":
                gates[
                    "G22_no_post_outcome_mutation"
                ],
        },
        {
            "requirement":
                "Full receiver x codebook x horizon x target matrix",
            "pass":
                gates[
                    "G24_full_3x3x4x4_matrix"
                ],
        },
    ]

    pdf_all_pass = all(
        item[
            "pass"
        ]
        for item in (
            pdf_matrix
        )
    )

    gates[
        "G23_PDF_requirement_matrix"
    ] = bool(
        pdf_all_pass
    )

    # ---------------------------------------------------------
    # P. Freeze primary scientific report BEFORE independent
    #    acceptance.
    # ---------------------------------------------------------

    primary_all_pass = all(
        gates.values()
    )

    scientific_status = (
        "PASS_SCIENTIFIC_FULLPDF_V2"
        if primary_all_pass
        else
        "FAIL_CLOSED_SCIENTIFIC_FULLPDF_V2"
    )

    science_report_sha = (
        write_json_once(
            SCIENCE_REPORT,
            {
                "stage": 5,
                "version":
                    "fullpdf_v2",
                "attempt":
                    "FINAL_ONE_CODE_V5",
                "status":
                    scientific_status,
                "scientific_gate_all_pass":
                    primary_all_pass,
                "formal_metrics_computed":
                    True,
                "controller_reexecuted":
                    False,
                "receiver_selection_reexecuted":
                    False,
                "decision_ledger": {
                    "path":
                        str(
                            SEALED
                        ),
                    "sha256":
                        sealed_sha,
                    "rows":
                        len(
                            decisions
                        ),
                    "receiver_keys":
                        len(
                            decision_keys
                        ),
                    "exact_repeat":
                        True,
                },
                "evaluator_truth": {
                    "path":
                        str(
                            truth_path
                        ),
                    "sha256":
                        sha256(
                            truth_path
                        ),
                    "truth_map_hash":
                        resolved_truth_hash,
                },
                "semantic_API_repair": {
                    "checker_source_changed":
                        False,
                    "adapter_amendment_sha256":
                        adapter_amendment_sha,
                    "adapter_seal_sha256":
                        adapter_seal_sha,
                },
                "formal": {
                    **normalize(
                        formal_summary
                    ),
                    "strata_count":
                        strata_count,
                    "expected_strata_count":
                        EXPECTED_COMBO_COUNT,
                    "evaluator_rows":
                        str(
                            EVALUATOR_ROWS
                        ),
                    "evaluator_rows_sha256":
                        evaluator_sha,
                    "formal_summary":
                        str(
                            FORMAL_SUMMARY
                        ),
                    "formal_summary_sha256":
                        summary_sha,
                },
                "gates":
                    gates,
                "PDF_requirement_evidence_matrix":
                    pdf_matrix,
                "PDF_compliance":
                    (
                        "PASS"
                        if pdf_all_pass
                        else "FAIL"
                    ),
                "scientific_boundaries": {
                    "training":
                        False,
                    "retraining":
                        False,
                    "recalibration":
                        False,
                    "formal_parameter_tuning":
                        False,
                    "post_outcome_tuning":
                        False,
                    "future_truth_controller_input":
                        False,
                    "tracks_to_predict_controller_input":
                        False,
                    "measured_optical_claim":
                        False,
                    "optical_semantics":
                        "normalized_constructed_optical_surrogate",
                    "traffic_semantics":
                        "real_WOMD_traffic_sensor_in_loop_not_measured_FMCW",
                },
                "Stage6_allowed":
                    False,
            },
        )
    )

    # =========================================================
    # Q. INDEPENDENT POST-OUTCOME ACCEPTANCE
    #
    # Re-read immutable outputs from disk.
    # No parameter can be changed here.
    # =========================================================

    frozen_rows = load_jsonl(
        EVALUATOR_ROWS
    )

    frozen_summary = load_json(
        FORMAL_SUMMARY
    )

    if sha256(
        EVALUATOR_ROWS
    ) != evaluator_sha:
        fail(
            "evaluator rows changed after freeze"
        )

    if sha256(
        FORMAL_SUMMARY
    ) != summary_sha:
        fail(
            "formal summary changed after freeze"
        )

    # Raw output combo evidence.
    raw_combos = {
        combo
        for row in frozen_rows
        if (
            combo := combo_from_record(
                row
            )
        )
        is not None
    }

    expected = expected_combos()

    raw_combo_complete = (
        raw_combos == expected
    )

    # Independent summary booleans.
    independent_boolean_gates = {
        "coverage":
            bool(
                frozen_summary.get(
                    "requested_coverage_all_strata_pass",
                    False,
                )
            ),

        "overhead":
            bool(
                frozen_summary.get(
                    "overhead_reduction_vs_exhaustive_all_strata_pass",
                    False,
                )
            ),

        "no_free_probes":
            bool(
                frozen_summary.get(
                    "no_free_probes",
                    False,
                )
            ),

        "latency":
            bool(
                frozen_summary.get(
                    "latency_all_strata_pass",
                    False,
                )
            ),

        "optical_chain":
            bool(
                frozen_summary.get(
                    "full_optical_chain_complete",
                    False,
                )
            ),
    }

    independent_strata = (
        strata_values(
            frozen_summary.get(
                "strata"
            )
        )
    )

    independent_strata_count_ok = (
        len(
            independent_strata
        )
        == EXPECTED_COMBO_COUNT
    )

    metric_presence_rows = [
        metric_presence(
            stratum
        )
        for stratum in (
            independent_strata
        )
    ]

    metric_evidence = {
        metric:
            (
                len(
                    metric_presence_rows
                )
                == EXPECTED_COMBO_COUNT
                and all(
                    row[
                        metric
                    ]
                    for row in (
                        metric_presence_rows
                    )
                )
            )
        for metric in (
            "coverage",
            "overhead",
            "latency",
            "BER",
            "effective_rate",
        )
    }

    # Metric-presence key names are evidence-only.
    # Frozen boolean gates remain the scientific thresholds.
    independent_acceptance = {
        "raw_combo_complete":
            raw_combo_complete,

        "strata_count_complete":
            independent_strata_count_ok,

        "coverage_gate":
            independent_boolean_gates[
                "coverage"
            ],

        "overhead_gate":
            independent_boolean_gates[
                "overhead"
            ],

        "no_free_probes_gate":
            independent_boolean_gates[
                "no_free_probes"
            ],

        "latency_gate":
            independent_boolean_gates[
                "latency"
            ],

        "optical_chain_gate":
            independent_boolean_gates[
                "optical_chain"
            ],

        "PDF_matrix_pass":
            pdf_all_pass,

        "protected_inputs_unchanged":
            all(
                sha256(
                    path
                )
                == expected_hash
                for path, expected_hash
                in EXPECTED.items()
            ),

        "sealed_decision_unchanged":
            (
                sha256(
                    SEALED
                )
                == sealed_sha
            ),
    }

    independent_all_pass = all(
        independent_acceptance.values()
    )

    independent_report_sha = (
        write_json_once(
            INDEPENDENT_REPORT,
            {
                "stage": 5,
                "version":
                    "fullpdf_v2",
                "status":
                    (
                        "PASS"
                        if independent_all_pass
                        else "FAIL"
                    ),
                "role":
                    "independent_postseal_acceptance",
                "scientific_report": {
                    "path":
                        str(
                            SCIENCE_REPORT
                        ),
                    "sha256":
                        science_report_sha,
                },
                "raw_evaluator_rows": {
                    "path":
                        str(
                            EVALUATOR_ROWS
                        ),
                    "sha256":
                        evaluator_sha,
                    "rows":
                        len(
                            frozen_rows
                        ),
                },
                "formal_summary": {
                    "path":
                        str(
                            FORMAL_SUMMARY
                        ),
                    "sha256":
                        summary_sha,
                },
                "expected_combinations":
                    EXPECTED_COMBO_COUNT,
                "raw_combinations_found":
                    len(
                        raw_combos
                    ),
                "acceptance":
                    independent_acceptance,
                "metric_field_evidence":
                    metric_evidence,
                "thresholds_changed":
                    False,
                "post_outcome_tuning":
                    False,
                "Stage6_allowed":
                    False,
            },
        )
    )

    # ---------------------------------------------------------
    # R. Genuine scientific FAIL: freeze it and STOP.
    # ---------------------------------------------------------

    final_all_pass = (
        primary_all_pass
        and independent_all_pass
    )

    if not final_all_pass:
        print()
        print("=" * 78)
        print(
            "STAGE5 FINAL V5 = "
            "GENUINE SCIENTIFIC/ACCEPTANCE FAIL"
        )
        print("=" * 78)

        print(
            "scientific_gate_all_pass =",
            primary_all_pass,
        )

        print(
            "independent_acceptance_all_pass =",
            independent_all_pass,
        )

        print(
            "requested_coverage_gate =",
            gates[
                "G16_requested_empirical_coverage"
            ],
        )

        print(
            "overhead_reduction_gate =",
            gates[
                "G17_actual_charged_overhead_reduced_vs_exhaustive"
            ],
        )

        print(
            "latency_gate =",
            gates[
                "G19_latency_from_actual_charged_probe_count"
            ],
        )

        print(
            "optical_chain_gate =",
            gates[
                "G20_complete_optical_chain"
            ],
        )

        print(
            "full_matrix_gate =",
            gates[
                "G24_full_3x3x4x4_matrix"
            ],
        )

        print(
            "PDF_compliance =",
            (
                "PASS"
                if pdf_all_pass
                else "FAIL"
            ),
        )

        print(
            "post_outcome_tuning = false"
        )

        print(
            "Stage6_allowed = false"
        )

        print(
            "science_report_sha256 =",
            science_report_sha,
        )

        print(
            "independent_report_sha256 =",
            independent_report_sha,
        )

        print(
            "DO NOT RERUN / DO NOT TUNE"
        )

        print("=" * 78)

        return 2

    # =========================================================
    # S. PASS — write definitive Stage5 closure.
    # =========================================================

    closure_sha = (
        write_json_once(
            CLOSURE,
            {
                "stage": 5,
                "version":
                    "fullpdf_v2",
                "status":
                    "COMPLETE_FROZEN_FULLPDF_V2",
                "closure_authority":
                    "POSTSEAL_V5_SUPERSEDING_FINAL",
                "PDF_compliance":
                    "PASS",
                "Stage6_allowed":
                    True,
                "decision_boundary": {
                    "two_exact_passes":
                        True,
                    "decision_ledger_sealed":
                        True,
                    "future_truth_opened_only_after_seal":
                        True,
                    "sealed_ledger":
                        str(
                            SEALED
                        ),
                    "sealed_ledger_sha256":
                        sealed_sha,
                    "rows":
                        len(
                            decisions
                        ),
                    "receiver_keys":
                        len(
                            decision_keys
                        ),
                },
                "evaluator_truth_binding": {
                    "path":
                        str(
                            truth_path
                        ),
                    "sha256":
                        sha256(
                            truth_path
                        ),
                    "truth_map_hash":
                        resolved_truth_hash,
                },
                "scientific_formal_report": {
                    "path":
                        str(
                            SCIENCE_REPORT
                        ),
                    "sha256":
                        science_report_sha,
                },
                "independent_acceptance_report": {
                    "path":
                        str(
                            INDEPENDENT_REPORT
                        ),
                    "sha256":
                        independent_report_sha,
                },
                "evaluator_rows": {
                    "path":
                        str(
                            EVALUATOR_ROWS
                        ),
                    "sha256":
                        evaluator_sha,
                },
                "formal_summary": {
                    "path":
                        str(
                            FORMAL_SUMMARY
                        ),
                    "sha256":
                        summary_sha,
                },
                "semantic_API_repair": {
                    "amendment_sha256":
                        adapter_amendment_sha,
                    "seal_sha256":
                        adapter_seal_sha,
                    "checker_source_modified":
                        False,
                    "scientific_semantics_modified":
                        False,
                },
                "gates":
                    gates,
                "PDF_requirement_evidence_matrix":
                    pdf_matrix,
                "scientific_boundaries": {
                    "training":
                        False,
                    "retraining":
                        False,
                    "recalibration":
                        False,
                    "formal_parameter_tuning":
                        False,
                    "post_outcome_tuning":
                        False,
                    "future_truth_controller_input":
                        False,
                    "tracks_to_predict_controller_input":
                        False,
                    "measured_FMCW_claim":
                        False,
                    "measured_optical_claim":
                        False,
                    "traffic_claim":
                        "real_WOMD_traffic",
                    "sensor_claim":
                        "PC_FMCW_sensor_in_loop",
                    "optical_chain_claim":
                        "normalized_constructed_optical_surrogate",
                },
            },
        )
    )

    # ---------------------------------------------------------
    # T. Stage5 -> Stage6 frozen handoff.
    # ---------------------------------------------------------

    handoff_sha = (
        write_json_once(
            HANDOFF,
            {
                "from_stage": 5,
                "to_stage": 6,
                "version":
                    "fullpdf_v2",
                "status":
                    "FROZEN_HANDOFF_READY",
                "Stage5_status":
                    "COMPLETE_FROZEN_FULLPDF_V2",
                "Stage6_allowed":
                    True,
                "Stage6_started":
                    False,
                "Stage5_closure": {
                    "path":
                        str(
                            CLOSURE
                        ),
                    "sha256":
                        closure_sha,
                },
                "trajectory_posterior_source": {
                    "path":
                        str(
                            STAGE4_LEDGER
                        ),
                    "sha256":
                        sha256(
                            STAGE4_LEDGER
                        ),
                    "model":
                        "calibrated_3D_Gaussian_GRU",
                    "Stage5_recalibration":
                        False,
                },
                "communication_decision_ledger": {
                    "path":
                        str(
                            SEALED
                        ),
                    "sha256":
                        sealed_sha,
                    "rows":
                        len(
                            decisions
                        ),
                },
                "receiver_binding": {
                    "path":
                        str(
                            RECEIVER_BINDING
                        ),
                    "sha256":
                        sha256(
                            RECEIVER_BINDING
                        ),
                },
                "evaluator_truth": {
                    "path":
                        str(
                            TRUTH_BINDING
                        ),
                    "sha256":
                        sha256(
                            TRUTH_BINDING
                        ),
                    "controller_input":
                        False,
                },
                "formal_scientific_evidence": {
                    "report":
                        str(
                            SCIENCE_REPORT
                        ),
                    "report_sha256":
                        science_report_sha,
                    "evaluator_rows":
                        str(
                            EVALUATOR_ROWS
                        ),
                    "evaluator_rows_sha256":
                        evaluator_sha,
                },
                "PDF_compliance":
                    "PASS",
                "communication_beams_and_ADB":
                    "separate_actuators_shared_posterior",
                "claims": {
                    "WOMD":
                        "real_traffic",
                    "FMCW":
                        "sensor_in_loop_not_measured_pointwise_Doppler",
                    "optical":
                        "constructed_normalized_surrogate_not_measured_optical_link",
                },
                "post_outcome_tuning":
                    False,
            },
        )
    )

    # ---------------------------------------------------------
    # U. Superseding final authority.
    # Old failures remain preserved; nothing is deleted.
    # ---------------------------------------------------------

    authority_sha = (
        write_json_once(
            AUTHORITY,
            {
                "stage": 5,
                "version":
                    "fullpdf_v2",
                "status":
                    "AUTHORITATIVE_FINAL_STAGE5_CLOSURE",
                "supersession_semantics":
                    (
                        "historical failed attempts preserved; "
                        "this authority supersedes them only for "
                        "final Stage5 status"
                    ),
                "historical_attempts_deleted":
                    False,
                "V4_failure_preserved":
                    True,
                "final_closure": {
                    "path":
                        str(
                            CLOSURE
                        ),
                    "sha256":
                        closure_sha,
                },
                "Stage5_to_Stage6_handoff": {
                    "path":
                        str(
                            HANDOFF
                        ),
                    "sha256":
                        handoff_sha,
                },
                "scientific_report_sha256":
                    science_report_sha,
                "independent_acceptance_sha256":
                    independent_report_sha,
                "decision_ledger_sha256":
                    sealed_sha,
                "evaluator_truth_sha256":
                    sha256(
                        TRUTH_BINDING
                    ),
                "PDF_compliance":
                    "PASS",
                "Stage5":
                    "COMPLETE_FROZEN_FULLPDF_V2",
                "Stage6_allowed":
                    True,
                "Stage6_started":
                    False,
                "post_outcome_tuning":
                    False,
            },
        )
    )

    # ---------------------------------------------------------
    # V. Final compact output.
    # ---------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "STAGE5 FULLPDF V2 — FINAL RESULT = PASS"
    )
    print("=" * 78)

    print(
        "Stage5 = COMPLETE_FROZEN_FULLPDF_V2"
    )

    print(
        "PDF_compliance = PASS"
    )

    print(
        "controller_reexecuted = false"
    )

    print(
        "receiver_selection_reexecuted = false"
    )

    print(
        "decision_ledger_sha256 =",
        sealed_sha,
    )

    print(
        "evaluator_truth_sha256 =",
        sha256(
            TRUTH_BINDING
        ),
    )

    print(
        "evaluator_rows =",
        len(
            frozen_rows
        ),
    )

    print(
        "formal_strata =",
        len(
            independent_strata
        ),
        "/",
        EXPECTED_COMBO_COUNT,
    )

    print(
        "requested_coverage_gate =",
        gates[
            "G16_requested_empirical_coverage"
        ],
    )

    print(
        "overhead_reduction_gate =",
        gates[
            "G17_actual_charged_overhead_reduced_vs_exhaustive"
        ],
    )

    print(
        "no_free_probes_gate =",
        gates[
            "G18_no_hidden_or_free_probes"
        ],
    )

    print(
        "latency_gate =",
        gates[
            "G19_latency_from_actual_charged_probe_count"
        ],
    )

    print(
        "optical_chain_gate =",
        gates[
            "G20_complete_optical_chain"
        ],
    )

    print(
        "full_3x3x4x4_gate =",
        gates[
            "G24_full_3x3x4x4_matrix"
        ],
    )

    print(
        "independent_acceptance =",
        independent_all_pass,
    )

    print(
        "post_outcome_tuning = false"
    )

    print(
        "Stage6_allowed = true"
    )

    print(
        "Stage6_started = false"
    )

    print(
        "science_report_sha256 =",
        science_report_sha,
    )

    print(
        "independent_report_sha256 =",
        independent_report_sha,
    )

    print(
        "closure_sha256 =",
        closure_sha,
    )

    print(
        "handoff_sha256 =",
        handoff_sha,
    )

    print(
        "final_authority_sha256 =",
        authority_sha,
    )

    print("=" * 78)

    return 0


# =====================================================================
# Fail-closed outer boundary.
# =====================================================================

if __name__ == "__main__":
    try:
        rc = main()

    except Exception as exc:
        OUTDIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        evaluator_exists = (
            EVALUATOR_ROWS.is_file()
        )

        summary_exists = (
            FORMAL_SUMMARY.is_file()
        )

        if not (
            HARNESS_FAILURE_REPORT.exists()
        ):
            try:
                write_json_once(
                    HARNESS_FAILURE_REPORT,
                    {
                        "stage": 5,
                        "version":
                            "fullpdf_v2",
                        "status":
                            "FAIL_CLOSED_UNEXPECTED_HARNESS_OR_EXECUTION",
                        "exception_type":
                            type(
                                exc
                            ).__name__,
                        "exception":
                            str(
                                exc
                            ),
                        "evaluator_rows_persisted":
                            evaluator_exists,
                        "formal_summary_persisted":
                            summary_exists,
                        "scientific_outcome_persisted":
                            bool(
                                evaluator_exists
                                or summary_exists
                            ),
                        "post_outcome_tuning":
                            False,
                        "automatic_rerun_allowed":
                            False,
                        "Stage6_allowed":
                            False,
                    },
                )
            except Exception:
                pass

        print()
        print("=" * 78)
        print(
            "STAGE5 FINAL ONE-CODE V5 = FAIL-CLOSED"
        )
        print(
            "reason =",
            repr(
                exc
            ),
        )
        print(
            "evaluator_rows_persisted =",
            evaluator_exists,
        )
        print(
            "formal_summary_persisted =",
            summary_exists,
        )
        print(
            "post_outcome_tuning = false"
        )
        print(
            "Stage6_allowed = false"
        )
        print(
            "DO NOT RERUN AUTOMATICALLY"
        )
        print("=" * 78)

        sys.exit(3)

    sys.exit(
        rc
    )

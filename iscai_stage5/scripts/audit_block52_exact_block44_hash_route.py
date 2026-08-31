from __future__ import annotations

import ast
from hashlib import sha256
import json
from pathlib import Path
import traceback


ROOT = Path("/home/agni/waymo")

STAGE4 = ROOT / "iscai_stage4"
STAGE5 = ROOT / "iscai_stage5"

TRAINING = (
    STAGE4
    / "scripts/run_block44_gaussian_training.py"
)

CONFIG = (
    STAGE4
    / "configs/stage4_gaussian_gru.json"
)

REPORT44 = (
    STAGE4
    / "reports/block44_gaussian_gru.json"
)

REPORT = (
    STAGE5
    / "artifacts/block52/"
      "exact_block44_hash_route_audit.json"
)


TARGET_FUNCTIONS = (
    "probabilistic_prediction_sha256",
    "evaluate_development",
    "build_gaussian_model",
)

TARGET_NAME_TOKENS = (
    "normal",
    "dataset",
    "batch",
    "development",
)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def file_sha256(path: Path):
    h = sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def write_json(path: Path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")

    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    tmp.replace(path)


def node_source(text, node):
    value = ast.get_source_segment(
        text,
        node,
    )

    if value is None:
        return None

    return value


def source_window(
    lines,
    line_number,
    *,
    before=10,
    after=15,
):
    start = max(
        1,
        int(line_number) - before,
    )

    end = min(
        len(lines),
        int(line_number) + after,
    )

    return {
        "start_line": start,
        "end_line": end,
        "lines": [
            {
                "line": index,
                "text": lines[index - 1],
            }
            for index in range(
                start,
                end + 1,
            )
        ],
    }


def main():
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.2 — EXACT BLOCK4.4 HASH/INFERENCE ROUTE AUDIT"
    )
    print(
        "============================================================"
    )

    required = (
        TRAINING,
        CONFIG,
        REPORT44,
    )

    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]

    require(
        not missing,
        "Missing frozen prerequisite(s): "
        + ", ".join(missing),
    )

    print("model forward executed = NO")
    print("formal N=120 used      = NO")
    print("dataset scan           = NO")
    print("training/recalibration = NO")
    print("Stage4 modified        = NO")

    text = TRAINING.read_text(
        encoding="utf-8"
    )

    lines = text.splitlines()

    tree = ast.parse(text)

    # --------------------------------------------------------
    # A. Exact function/class definitions
    # --------------------------------------------------------

    print()
    print(
        "============================================================"
    )
    print(
        "A. EXACT BLOCK4.4 FUNCTION / CLASS DEFINITIONS"
    )
    print(
        "============================================================"
    )

    definitions = {}

    for node in tree.body:

        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            name = node.name

            interesting = (
                name
                in TARGET_FUNCTIONS
                or
                any(
                    token in name.lower()
                    for token in TARGET_NAME_TOKENS
                )
            )

            if not interesting:
                continue

            source = node_source(
                text,
                node,
            )

            definitions[name] = {
                "kind":
                    type(node).__name__,

                "line":
                    int(node.lineno),

                "end_line":
                    int(
                        getattr(
                            node,
                            "end_lineno",
                            node.lineno,
                        )
                    ),

                "source":
                    source,
            }

    for name, item in definitions.items():

        print()
        print(
            "------------------------------------------------------------"
        )
        print(
            name,
            f"L{item['line']}-L{item['end_line']}",
        )
        print(
            "------------------------------------------------------------"
        )

        source = item.get("source")

        if source:
            print(source[:16000])

    require(
        "probabilistic_prediction_sha256"
        in definitions,
        (
            "Could not resolve frozen "
            "probabilistic_prediction_sha256."
        ),
    )

    # --------------------------------------------------------
    # B. All calls to the frozen hash function
    # --------------------------------------------------------

    print()
    print(
        "============================================================"
    )
    print(
        "B. probabilistic_prediction_sha256 CALL SITES"
    )
    print(
        "============================================================"
    )

    hash_calls = []

    for node in ast.walk(tree):

        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        function = node.func

        name = None

        if isinstance(
            function,
            ast.Name,
        ):
            name = function.id

        elif isinstance(
            function,
            ast.Attribute,
        ):
            name = function.attr

        if (
            name
            !=
            "probabilistic_prediction_sha256"
        ):
            continue

        source = node_source(
            text,
            node,
        )

        item = {
            "line":
                int(node.lineno),

            "call_source":
                source,

            "window":
                source_window(
                    lines,
                    node.lineno,
                    before=18,
                    after=24,
                ),
        }

        hash_calls.append(item)

        print()
        print(
            "CALL AT LINE",
            node.lineno,
        )

        print(
            "call =",
            source,
        )

        for row in item[
            "window"
        ][
            "lines"
        ]:
            print(
                f"{row['line']:5d}: "
                f"{row['text']}"
            )

    require(
        hash_calls,
        (
            "No frozen prediction-hash "
            "call site found."
        ),
    )

    # --------------------------------------------------------
    # C. Exact development evaluation route around model()
    # --------------------------------------------------------

    print()
    print(
        "============================================================"
    )
    print(
        "C. DEVELOPMENT FORWARD / DENORMALIZATION ROUTE"
    )
    print(
        "============================================================"
    )

    route_tokens = (
        "model(",
        "denormalize_gaussian",
        "covariance_from_scale_tril",
        "development_loader",
        "dev_loader",
        "DEV_CACHE",
        "batch_size",
        "DataLoader",
        "prediction_sha256",
        "probabilistic_prediction_sha256",
        "torch.inference_mode",
        "torch.no_grad",
    )

    route_hits = []

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        lower = line.lower()

        if not any(
            token.lower() in lower
            for token in route_tokens
        ):
            continue

        route_hits.append({
            "line":
                line_number,

            "text":
                line,
        })

    for hit in route_hits:
        print(
            f"L{hit['line']:4d}: "
            f"{hit['text']}"
        )

    # --------------------------------------------------------
    # D. Exact determinism/device setup
    # --------------------------------------------------------

    print()
    print(
        "============================================================"
    )
    print(
        "D. DETERMINISM / DEVICE / SEED ROUTE"
    )
    print(
        "============================================================"
    )

    determinism_tokens = (
        "set_global_determinism",
        "manual_seed",
        "cuda",
        "cudnn",
        "determin",
        "CUBLAS",
        "device",
        "seed",
        "float32",
        "float64",
        "matmul",
        "tf32",
    )

    determinism_hits = []

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        lower = line.lower()

        if not any(
            token.lower() in lower
            for token in determinism_tokens
        ):
            continue

        determinism_hits.append({
            "line":
                line_number,

            "text":
                line,
        })

    for hit in determinism_hits[
        :180
    ]:
        print(
            f"L{hit['line']:4d}: "
            f"{hit['text']}"
        )

    # --------------------------------------------------------
    # E. Frozen config/report hash evidence
    # --------------------------------------------------------

    print()
    print(
        "============================================================"
    )
    print(
        "E. FROZEN HASH / CONFIG EVIDENCE"
    )
    print(
        "============================================================"
    )

    config = json.loads(
        CONFIG.read_text(
            encoding="utf-8"
        )
    )

    report44 = json.loads(
        REPORT44.read_text(
            encoding="utf-8"
        )
    )

    expected_hash = (
        report44[
            "checkpoint"
        ][
            "prediction_sha256"
        ]
    )

    print(
        "training script SHA256 =",
        file_sha256(TRAINING),
    )

    print(
        "config SHA256 =",
        file_sha256(CONFIG),
    )

    print(
        "frozen prediction SHA256 =",
        expected_hash,
    )

    training = config.get(
        "training",
        {}
    )

    print(
        "config training =",
        json.dumps(
            training,
            indent=2,
            sort_keys=True,
            default=str,
        ),
    )

    # --------------------------------------------------------
    # F. Decision
    # --------------------------------------------------------

    hash_definition_source = (
        definitions[
            "probabilistic_prediction_sha256"
        ][
            "source"
        ]
        or
        ""
    )

    uses_runtime_hash_name = (
        "prediction_sha256("
        in hash_definition_source
    )

    uses_denormalized_name = any(
        token in hash_definition_source
        for token in (
            "denormal",
            "metric_mean",
            "metric_scale",
            "covariance",
        )
    )

    print()
    print(
        "============================================================"
    )
    print(
        "F. FORENSIC DECISION"
    )
    print(
        "============================================================"
    )

    print(
        "frozen Block4.4 hash function = FOUND"
    )

    print(
        "hash call sites              =",
        len(hash_calls),
    )

    print(
        "calls runtime prediction_sha256 =",
        uses_runtime_hash_name,
    )

    print(
        "hash source mentions metric/denormalized values =",
        uses_denormalized_name,
    )

    print(
        "batch-size hypothesis alone  = REJECTED"
    )

    print(
        "hash gate relaxation          = FORBIDDEN"
    )

    print(
        "next = reproduce exact frozen "
        "Block4.4 hash function + exact preprocessing"
    )

    payload = {
        "status":
            "PASS_READ_ONLY_FORENSIC_AUDIT",

        "scientific_execution": {
            "model_forward":
                False,

            "formal_N120_read":
                False,

            "dataset_scan":
                False,

            "training":
                False,

            "recalibration":
                False,

            "Stage4_modified":
                False,
        },

        "training_script": {
            "path":
                str(TRAINING),

            "sha256":
                file_sha256(TRAINING),
        },

        "definitions":
            definitions,

        "prediction_hash_calls":
            hash_calls,

        "development_route_hits":
            route_hits,

        "determinism_hits":
            determinism_hits,

        "frozen_prediction_sha256":
            expected_hash,

        "configuration_training":
            training,

        "decision": {
            "batch_size_only_explanation":
                False,

            "relax_exact_hash_gate":
                False,

            "next":
                (
                    "reproduce_exact_Block44_"
                    "probabilistic_prediction_sha256_"
                    "and_preprocessing_route"
                ),
        },
    }

    write_json(
        REPORT,
        payload,
    )

    print()
    print(
        "STATUS = PASS_READ_ONLY_AUDIT"
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "model forward executed = NO"
    )

    print(
        "formal N=120 used = NO"
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
        "EXACT BLOCK4.4 HASH ROUTE AUDIT = BLOCKED"
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

    print(
        "model forward executed = NO"
    )

    print(
        "formal N=120 used = NO"
    )

    print(
        "training/recalibration = NO"
    )

    print(
        "Stage4 modified = NO"
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no sys.exit().

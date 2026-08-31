from __future__ import annotations

import importlib
import inspect
import json
from hashlib import sha256
from pathlib import Path
import traceback


ROOT = Path(
    "/home/agni/waymo"
)

STAGE4 = (
    ROOT
    / "iscai_stage4"
)

STAGE5 = (
    ROOT
    / "iscai_stage5"
)

HANDOFF = (
    STAGE4
    / "artifacts/block410/"
      "stage4_to_stage5_handoff.json"
)

RUNTIME_SOURCE = (
    STAGE4
    / "src/iscai_stage4/ml/"
      "calibration_runtime.py"
)

REPORT = (
    STAGE5
    / "artifacts/block52/"
      "calibrated_runtime_api_audit.json"
)

MODULE_NAME = (
    "iscai_stage4.ml.calibration_runtime"
)

KEYWORDS = (
    "calibr",
    "runtime",
    "predict",
    "gaussian",
    "load",
    "checkpoint",
    "normal",
    "covariance",
    "displacement",
    "origin",
    "anchor",
    "mean",
)


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


def file_sha256(
    path: Path,
):
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
            default=str,
        )
        +
        "\n",
        encoding="utf-8",
    )

    temporary.replace(
        path
    )


def safe_signature(
    obj,
):
    try:
        return str(
            inspect.signature(
                obj
            )
        )

    except Exception:
        return None


def callable_record(
    name,
    obj,
):
    try:
        source_file = (
            inspect.getsourcefile(
                obj
            )
        )

    except Exception:
        source_file = None

    try:
        source_lines, line_number = (
            inspect.getsourcelines(
                obj
            )
        )

        source_preview = "".join(
            source_lines[
                :80
            ]
        )[
            :12000
        ]

    except Exception:
        line_number = None
        source_preview = None

    return {
        "name":
            name,

        "kind":
            (
                "class"
                if inspect.isclass(
                    obj
                )
                else
                "function"
            ),

        "signature":
            safe_signature(
                obj
            ),

        "module":
            getattr(
                obj,
                "__module__",
                None,
            ),

        "source_file":
            source_file,

        "definition_line":
            line_number,

        "source_preview":
            source_preview,
    }


def relevant(
    name,
    record,
):
    haystack = (
        str(
            name
        )
        +
        "\n"
        +
        str(
            record.get(
                "signature"
            )
        )
        +
        "\n"
        +
        str(
            record.get(
                "source_preview"
            )
        )
    ).lower()

    return any(
        keyword in haystack
        for keyword in KEYWORDS
    )


def source_hits():
    text = RUNTIME_SOURCE.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    rows = []

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        lower = line.lower()

        if not any(
            keyword in lower
            for keyword in KEYWORDS
        ):
            continue

        rows.append({
            "line":
                line_number,

            "text":
                line[
                    :600
                ],
        })

    return rows


def flatten_json(
    value,
    prefix="",
):
    rows = []

    if isinstance(
        value,
        dict,
    ):
        for key, item in value.items():

            child = (
                f"{prefix}.{key}"
                if prefix
                else str(
                    key
                )
            )

            rows.extend(
                flatten_json(
                    item,
                    child,
                )
            )

    elif isinstance(
        value,
        list,
    ):
        if len(
            value
        ) <= 30:

            for index, item in enumerate(
                value
            ):
                rows.extend(
                    flatten_json(
                        item,
                        f"{prefix}[{index}]",
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


def main():
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.2 — CANONICAL CALIBRATED-RUNTIME API AUDIT"
    )
    print(
        "============================================================"
    )

    required = (
        HANDOFF,
        RUNTIME_SOURCE,
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
            "Missing frozen Stage4 prerequisite(s): "
            +
            ", ".join(
                missing
            )
        ),
    )

    print(
        "model forward pass    = NO"
    )

    print(
        "formal N=120 read     = NO"
    )

    print(
        "dataset scan          = NO"
    )

    print(
        "training/recalibration= NO"
    )

    print(
        "upstream modification = NO"
    )

    # ========================================================
    # A. Stage4→Stage5 handoff semantics
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "A. STAGE4→STAGE5 HANDOFF"
    )
    print(
        "============================================================"
    )

    handoff = json.loads(
        HANDOFF.read_text(
            encoding="utf-8"
        )
    )

    print(
        "handoff SHA256 =",
        file_sha256(
            HANDOFF
        ),
    )

    interesting_handoff = []

    for key, value in flatten_json(
        handoff
    ):
        lower = key.lower()

        if any(
            token in lower
            for token in (
                "default",
                "gaussian",
                "calibr",
                "checkpoint",
                "normal",
                "covariance",
                "runtime",
                "posterior",
                "horizon",
            )
        ):
            interesting_handoff.append({
                "key":
                    key,

                "value":
                    value,
            })

    for item in interesting_handoff[
        :100
    ]:
        print(
            item[
                "key"
            ],
            "=",
            item[
                "value"
            ],
        )

    # ========================================================
    # B. Runtime module import / public API
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "B. CALIBRATION_RUNTIME PUBLIC API"
    )
    print(
        "============================================================"
    )

    module = importlib.import_module(
        MODULE_NAME
    )

    module_path = Path(
        module.__file__
    ).resolve()

    require(
        module_path
        ==
        RUNTIME_SOURCE.resolve(),
        (
            "Imported calibration_runtime "
            "does not match frozen Stage4 source."
        ),
    )

    print(
        "module =",
        MODULE_NAME,
    )

    print(
        "source =",
        module_path,
    )

    print(
        "source SHA256 =",
        file_sha256(
            module_path
        ),
    )

    records = []

    for name in sorted(
        dir(
            module
        )
    ):
        if name.startswith(
            "_"
        ):
            continue

        obj = getattr(
            module,
            name
        )

        if not (
            inspect.isfunction(
                obj
            )
            or
            inspect.isclass(
                obj
            )
        ):
            continue

        #
        # Only definitions belonging to Stage4 code,
        # not imported torch/pathlib classes.
        #
        obj_module = getattr(
            obj,
            "__module__",
            ""
        )

        if not str(
            obj_module
        ).startswith(
            "iscai_stage4"
        ):
            continue

        record = callable_record(
            name,
            obj,
        )

        records.append(
            record
        )

        print()
        print(
            name,
            "|",
            record[
                "kind"
            ],
        )

        print(
            "  signature =",
            record[
                "signature"
            ],
        )

        print(
            "  source =",
            record[
                "source_file"
            ],
        )

        print(
            "  line =",
            record[
                "definition_line"
            ],
        )

    # ========================================================
    # C. Relevant callable bodies
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "C. RELEVANT RUNTIME CALLABLE DEFINITIONS"
    )
    print(
        "============================================================"
    )

    relevant_records = [
        record
        for record in records
        if relevant(
            record[
                "name"
            ],
            record,
        )
    ]

    for record in relevant_records:

        print()
        print(
            "------------------------------------------------------------"
        )

        print(
            record[
                "name"
            ],
            record[
                "signature"
            ],
        )

        print(
            "------------------------------------------------------------"
        )

        preview = record.get(
            "source_preview"
        )

        if preview:
            print(
                preview
            )

    # ========================================================
    # D. Raw source hits
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "D. CALIBRATION_RUNTIME CRITICAL SOURCE HITS"
    )
    print(
        "============================================================"
    )

    hits = source_hits()

    for hit in hits[
        :160
    ]:
        print(
            f"L{hit['line']:4d}: "
            f"{hit['text']}"
        )

    # ========================================================
    # E. Decision gate
    # ========================================================

    names = {
        record[
            "name"
        ]
        for record in records
    }

    print()
    print(
        "============================================================"
    )
    print(
        "E. RUNTIME-ROUTE DECISION"
    )
    print(
        "============================================================"
    )

    print(
        "public Stage4 callables =",
        len(
            records
        ),
    )

    print(
        "relevant callables      =",
        len(
            relevant_records
        ),
    )

    print(
        "callable names          =",
        sorted(
            names
        ),
    )

    print(
        "absolute-position rule  = "
        "current causal H0 anchor + predicted displacement"
    )

    print(
        "translation covariance  = UNCHANGED"
    )

    print(
        "manual Stage4 inference reimplementation approved = NO"
    )

    print(
        "canonical runtime reuse preferred = YES"
    )

    report_payload = {
        "status":
            "PASS_READ_ONLY_RUNTIME_AUDIT",

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

            "upstream_modified":
                False,
        },

        "handoff": {
            "path":
                str(
                    HANDOFF
                ),

            "sha256":
                file_sha256(
                    HANDOFF
                ),

            "interesting_fields":
                interesting_handoff,
        },

        "runtime": {
            "module":
                MODULE_NAME,

            "source":
                str(
                    module_path
                ),

            "source_sha256":
                file_sha256(
                    module_path
                ),

            "public_callables":
                records,

            "relevant_callables":
                relevant_records,

            "critical_source_hits":
                hits,
        },

        "coordinate_semantics": {
            "Stage4_prediction":
                "H0_displacement",

            "absolute_H0_mean":
                (
                    "latest_causal_observed_"
                    "position_H0_plus_"
                    "predicted_displacement"
                ),

            "translation_changes_covariance":
                False,
        },

        "next_decision":
            (
                "reuse_canonical_calibrated_runtime_"
                "if_prediction_API_is_exposed_else_"
                "reproduce_exact_frozen_Block44_route"
            ),
    }

    write_json(
        REPORT,
        report_payload,
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
        "CALIBRATED-RUNTIME API AUDIT = BLOCKED"
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

    print(
        "model forward executed = NO"
    )

    print(
        "formal N=120 used      = NO"
    )

    print(
        "training/recalibration = NO"
    )

    print(
        "upstream modified      = NO"
    )

    print(
        "terminal remains open  = YES"
    )

# Deliberately no sys.exit().

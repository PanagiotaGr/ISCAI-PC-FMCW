from __future__ import annotations

import ast
from dataclasses import fields, is_dataclass
from hashlib import sha256
import inspect
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

MODULE_PATH = (
    S6
    / "src/iscai_stage6/adb/"
      "class_aware_policy.py"
)

RECON = (
    S6
    / "reports/"
      "block68_preoutcome_metric_reconciliation.json"
)

ROUTE_BIND = (
    S6
    / "reports/"
      "block68_part2_development_route_schema_bind.json"
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

REPORT = (
    S6
    / "reports/"
      "block68_final_illumination_route_extract.json"
)

EXPECTED_RECON_SHA = (
    "077a89fa5163f8862a3a5b7489d98929"
    "d9ae3e160ed9e030eb75f42a91d48f9f"
)

EXPECTED_PROTOCOL_SHA = (
    "409fbb2785e4ff13f29fdff96c91d608"
    "c245e009fcbb86ce57099d6a4e1815c9"
)

EXPECTED_METRIC_SHA = (
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

    require(
        "formal"
        not in
        str(
            path.resolve()
        ).lower(),
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


def source_excerpt(
    function,
    *,
    max_lines=160,
):

    try:

        source = inspect.getsource(
            function
        )

    except Exception as exc:

        return [
            (
                "<source unavailable: "
                f"{type(exc).__name__}: {exc}>"
            )
        ]

    lines = (
        source.splitlines()
    )

    return lines[
        :max_lines
    ]


def annotation_text(
    annotation,
):

    if annotation is inspect.Signature.empty:
        return "<none>"

    return str(
        annotation
    )


# ============================================================
# AST route extraction
# ============================================================

def ast_function_routes(
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

    interesting_tokens = (
        "illum",
        "temporal",
        "smooth",
        "schedule",
        "rate",
        "class",
        "compose",
        "mask",
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

        lower_name = (
            node.name.lower()
        )

        if not any(
            token in lower_name
            for token in interesting_tokens
        ):
            continue

        calls = []
        return_names = []
        constructor_keywords = []

        for child in ast.walk(
            node
        ):

            if isinstance(
                child,
                ast.Call,
            ):

                if isinstance(
                    child.func,
                    ast.Name,
                ):

                    call_name = (
                        child.func.id
                    )

                elif isinstance(
                    child.func,
                    ast.Attribute,
                ):

                    call_name = (
                        child.func.attr
                    )

                else:

                    call_name = (
                        "<complex>"
                    )

                calls.append(
                    call_name
                )

                keywords = [
                    keyword.arg
                    for keyword in child.keywords
                    if keyword.arg
                    is not None
                ]

                if keywords:

                    constructor_keywords.append({
                        "call":
                            call_name,

                        "keywords":
                            keywords,
                    })

            elif isinstance(
                child,
                ast.Return,
            ):

                value = child.value

                if isinstance(
                    value,
                    ast.Name,
                ):

                    return_names.append(
                        value.id
                    )

                elif isinstance(
                    value,
                    ast.Call,
                ):

                    if isinstance(
                        value.func,
                        ast.Name,
                    ):

                        return_names.append(
                            (
                                "call:"
                                +
                                value.func.id
                            )
                        )

                    elif isinstance(
                        value.func,
                        ast.Attribute,
                    ):

                        return_names.append(
                            (
                                "call:"
                                +
                                value.func.attr
                            )
                        )

        functions.append({
            "name":
                node.name,

            "line":
                node.lineno,

            "calls":
                sorted(
                    set(
                        calls
                    )
                ),

            "returns":
                sorted(
                    set(
                        return_names
                    )
                ),

            "constructor_keywords":
                constructor_keywords,
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
        "FINAL ILLUMINATION ROUTE EXTRACTION"
    )
    print(
        "READ-ONLY SCIENTIFIC DIAGNOSIS"
    )
    print(
        "============================================================"
    )

    required = (
        MODULE_PATH,
        RECON,
        ROUTE_BIND,
        PROTOCOL,
        METRICS,
        INTERFACE,
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
            "Missing required evidence: "
            +
            ", ".join(
                missing
            )
        ),
    )

    # ========================================================
    # A. Frozen seals
    # ========================================================

    print()
    print(
        "===== A. FROZEN SEALS ====="
    )

    require(
        sha256_file(
            RECON
        )
        ==
        EXPECTED_RECON_SHA,
        (
            "Reconciliation SHA changed."
        ),
    )

    require(
        sha256_file(
            PROTOCOL
        )
        ==
        EXPECTED_PROTOCOL_SHA,
        (
            "Protocol SHA changed."
        ),
    )

    require(
        sha256_file(
            METRICS
        )
        ==
        EXPECTED_METRIC_SHA,
        (
            "Metric runtime SHA changed."
        ),
    )

    require(
        sha256_file(
            INTERFACE
        )
        ==
        EXPECTED_INTERFACE_SHA,
        (
            "Metric interface SHA changed."
        ),
    )

    route = load_json(
        ROUTE_BIND
    )

    require(
        route.get(
            "status"
        )
        ==
        "PASS_DEVELOPMENT_ROUTE_SCHEMA_BOUND",
        (
            "Previous route bind not PASS."
        ),
    )

    before_module_sha = (
        sha256_file(
            MODULE_PATH
        )
    )

    print(
        "reconciliation       = EXACT PASS"
    )

    print(
        "metric protocol      = EXACT PASS"
    )

    print(
        "metric runtime       = EXACT PASS"
    )

    print(
        "metric interface     = EXACT PASS"
    )

    print(
        "development route    = PASS"
    )

    # ========================================================
    # B. Module-wide dataclass inventory
    # ========================================================

    print()
    print(
        "===== B. CLASS-AWARE MODULE DATACLASSES ====="
    )

    import iscai_stage6.adb.class_aware_policy as cap

    dataclasses = {}

    for name, obj in inspect.getmembers(
        cap,
        inspect.isclass,
    ):

        if (
            getattr(
                obj,
                "__module__",
                None,
            )
            !=
            cap.__name__
        ):
            continue

        if not is_dataclass(
            obj
        ):
            continue

        dataclasses[
            name
        ] = [
            {
                "name":
                    field.name,

                "type":
                    str(
                        field.type
                    ),
            }
            for field in fields(
                obj
            )
        ]

        print()
        print(
            name
        )

        for field in fields(
            obj
        ):

            print(
                "   ",
                field.name,
                ":",
                field.type,
            )

    require(
        "ClassAwareComposition"
        in
        dataclasses,
        (
            "ClassAwareComposition missing."
        ),
    )

    composition_fields = {
        item[
            "name"
        ]
        for item in (
            dataclasses[
                "ClassAwareComposition"
            ]
        )
    }

    require(
        "raw_class_aware_illumination"
        in
        composition_fields,
        (
            "Expected raw class-aware "
            "illumination field missing."
        ),
    )

    require(
        "illumination"
        not in
        composition_fields,
        (
            "Unexpectedly found direct "
            "ClassAwareComposition.illumination."
        ),
    )

    print()
    print(
        "ClassAwareComposition.illumination = ABSENT CONFIRMED"
    )

    print(
        "ClassAwareComposition.raw_class_aware_illumination = PRESENT"
    )

    # ========================================================
    # C. Exact function inventory
    # ========================================================

    print()
    print(
        "===== C. RELEVANT FUNCTION SIGNATURES ====="
    )

    functions = {}

    name_tokens = (
        "illum",
        "temporal",
        "smooth",
        "schedule",
        "rate",
        "compose",
        "mask",
    )

    for name, obj in inspect.getmembers(
        cap,
        inspect.isfunction,
    ):

        if (
            getattr(
                obj,
                "__module__",
                None,
            )
            !=
            cap.__name__
        ):
            continue

        if not any(
            token
            in
            name.lower()
            for token in name_tokens
        ):
            continue

        signature = inspect.signature(
            obj
        )

        functions[
            name
        ] = {
            "signature":
                str(
                    signature
                ),

            "return_annotation":
                annotation_text(
                    signature.return_annotation
                ),
        }

        print()
        print(
            name
        )

        print(
            "   signature =",
            signature,
        )

        print(
            "   return     =",
            annotation_text(
                signature.return_annotation
            ),
        )

    require(
        "compose_class_aware_illumination"
        in
        functions,
        (
            "compose_class_aware_illumination "
            "missing."
        ),
    )

    require(
        "temporal_smooth_schedule"
        in
        functions,
        (
            "temporal_smooth_schedule missing."
        ),
    )

    # ========================================================
    # D. Exact source excerpts
    # ========================================================

    print()
    print(
        "===== D. EXACT PIPELINE SOURCE EXCERPTS ====="
    )

    preferred = (
        "compose_class_aware_illumination",
        "temporal_smooth_schedule",
    )

    extra_names = [
        name
        for name in functions
        if (
            "rate"
            in name.lower()
            or
            "schedule"
            in name.lower()
        )
        and
        name
        not in preferred
    ]

    excerpt_names = (
        list(
            preferred
        )
        +
        sorted(
            extra_names
        )
    )

    excerpts = {}

    for name in excerpt_names:

        if not hasattr(
            cap,
            name,
        ):
            continue

        function = getattr(
            cap,
            name,
        )

        lines = source_excerpt(
            function
        )

        excerpts[
            name
        ] = lines

        print()
        print(
            "------------------------------------------------------------"
        )
        print(
            "FUNCTION:",
            name,
        )
        print(
            "------------------------------------------------------------"
        )

        for line in lines:

            print(
                line
            )

    # ========================================================
    # E. AST producer/consumer chain
    # ========================================================

    print()
    print(
        "===== E. AST PRODUCER / CONSUMER CHAIN ====="
    )

    routes = ast_function_routes(
        MODULE_PATH
    )

    for item in routes:

        print()
        print(
            f"L{item['line']:04d} "
            f"{item['name']}"
        )

        print(
            "   calls =",
            item[
                "calls"
            ],
        )

        print(
            "   returns =",
            item[
                "returns"
            ],
        )

        important_constructors = [
            entry
            for entry in (
                item[
                    "constructor_keywords"
                ]
            )
            if any(
                key
                in
                (
                    "illumination",
                    "illumination_minimum",
                    "raw_class_aware_illumination",
                    "vru_floor_guard",
                )
                for key in (
                    entry[
                        "keywords"
                    ]
                )
            )
        ]

        if important_constructors:

            print(
                "   illumination constructors =",
                important_constructors,
            )

    # ========================================================
    # F. Find exact producers of keyword `illumination=`
    # ========================================================

    print()
    print(
        "===== F. `illumination=` PRODUCER SEARCH ====="
    )

    illumination_producers = []

    for item in routes:

        for constructor in (
            item[
                "constructor_keywords"
            ]
        ):

            if (
                "illumination"
                in
                constructor[
                    "keywords"
                ]
            ):

                record = {
                    "function":
                        item[
                            "name"
                        ],

                    "line":
                        item[
                            "line"
                        ],

                    "constructor":
                        constructor[
                            "call"
                        ],

                    "keywords":
                        constructor[
                            "keywords"
                        ],
                }

                illumination_producers.append(
                    record
                )

                print(
                    "producer function =",
                    record[
                        "function"
                    ],
                )

                print(
                    "constructor       =",
                    record[
                        "constructor"
                    ],
                )

                print(
                    "keywords          =",
                    record[
                        "keywords"
                    ],
                )

                print()

    # This is the critical thing our previous binder failed
    # to discover.
    require(
        illumination_producers,
        (
            "No runtime producer with "
            "`illumination=` was found."
        ),
    )

    # ========================================================
    # G. Determine exact final-route candidates
    # ========================================================

    print()
    print(
        "===== G. FINAL I_FINAL ROUTE CANDIDATES ====="
    )

    exact_field_candidates = []

    for class_name, field_items in (
        dataclasses.items()
    ):

        field_names = {
            item[
                "name"
            ]
            for item in field_items
        }

        if "illumination" in field_names:

            exact_field_candidates.append(
                class_name
            )

    ndarray_schedule_candidates = []

    for name, metadata in (
        functions.items()
    ):

        return_text = (
            metadata[
                "return_annotation"
            ].lower()
        )

        if (
            (
                "ndarray"
                in
                return_text
            )
            and
            any(
                token
                in
                name.lower()
                for token in (
                    "temporal",
                    "schedule",
                    "rate",
                )
            )
        ):

            ndarray_schedule_candidates.append(
                name
            )

    print(
        "dataclasses with exact `illumination` field =",
        exact_field_candidates,
    )

    print(
        "schedule/rate ndarray-return functions =",
        ndarray_schedule_candidates,
    )

    print(
        "illumination constructor producers =",
        illumination_producers,
    )

    require(
        (
            exact_field_candidates
            or
            ndarray_schedule_candidates
        ),
        (
            "No evidence-supported final "
            "illumination route candidate found."
        ),
    )

    # ========================================================
    # H. Scientific boundary
    # ========================================================

    print()
    print(
        "===== H. SCIENTIFIC BOUNDARY ====="
    )

    print(
        "development records read      = NO"
    )

    print(
        "development metrics computed  = NO"
    )

    print(
        "numeric bounds selected       = NO"
    )

    print(
        "formal content opened         = NO"
    )

    print(
        "formal outcomes read          = NO"
    )

    print(
        "policy tuning                 = NO"
    )

    print(
        "scientific source modified    = NO"
    )

    # ========================================================
    # I. Full regression
    # ========================================================

    print()
    print(
        "===== I. FULL STAGE6 REGRESSION ====="
    )

    rc, count, output = (
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
        count
        ==
        EXPECTED_STAGE6_TESTS,
        (
            f"Expected {EXPECTED_STAGE6_TESTS} "
            f"tests; got {count}."
        ),
    )

    print(
        "Stage6 regression =",
        f"{count} / {count} PASS",
    )

    # ========================================================
    # J. Immutability
    # ========================================================

    print()
    print(
        "===== J. IMMUTABILITY ====="
    )

    after_module_sha = (
        sha256_file(
            MODULE_PATH
        )
    )

    require(
        after_module_sha
        ==
        before_module_sha,
        (
            "class_aware_policy.py changed."
        ),
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
            "Metric protocol changed."
        ),
    )

    require(
        sha256_file(
            METRICS
        )
        ==
        EXPECTED_METRIC_SHA,
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
        "class_aware_policy.py = UNCHANGED"
    )

    print(
        "reconciled metrics     = UNCHANGED"
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
    # L. Route-extraction report
    # ========================================================

    result = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.8",

        "part":
            "final_illumination_route_extract",

        "status":
            "PASS_FINAL_ILLUMINATION_ROUTE_EXTRACTED",

        "diagnosis": {
            "previous_binder_error":
                (
                    "incorrectly required "
                    "ClassAwareComposition.illumination"
                ),

            "ClassAwareComposition_has_illumination":
                False,

            "ClassAwareComposition_has_raw_class_aware_illumination":
                True,

            "scientific_bug":
                False,

            "evidence_route_bug":
                True,
        },

        "dataclasses":
            dataclasses,

        "functions":
            functions,

        "AST_routes":
            routes,

        "illumination_producers":
            illumination_producers,

        "final_route_candidates": {
            "dataclasses_with_exact_illumination_field":
                exact_field_candidates,

            "schedule_ndarray_return_functions":
                ndarray_schedule_candidates,
        },

        "scientific_execution": {
            "development_records_read":
                False,

            "development_metrics_computed":
                False,

            "numeric_bounds_selected":
                False,

            "formal_content_opened":
                False,

            "formal_outcomes_read":
                False,

            "policy_tuning":
                False,

            "scientific_source_modified":
                False,
        },

        "regression": {
            "tests":
                count,

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
                "Repair only the Block6.8 exact evaluator "
                "binder to use the evidence-supported "
                "post-composition final illumination route."
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
        "BLOCK 6.8 FINAL ILLUMINATION ROUTE — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "previous binder assumption  = FALSE"
    )

    print(
        "scientific implementation   = UNCHANGED"
    )

    print(
        "ClassAwareComposition raw I = PROVEN"
    )

    print(
        "post-composition producers  =",
        len(
            illumination_producers
        ),
    )

    print(
        "final-field candidates      =",
        exact_field_candidates,
    )

    print(
        "schedule-array candidates   =",
        ndarray_schedule_candidates,
    )

    print(
        "development metrics         = NOT COMPUTED"
    )

    print(
        "numeric bounds              = NOT SELECTED"
    )

    print(
        "formal content              = NOT OPENED"
    )

    print(
        "policy tuning               = NO"
    )

    print(
        "Stage6 regression           =",
        f"{count} / {count} PASS",
    )

    print(
        "STATUS = PASS_FINAL_ILLUMINATION_ROUTE_EXTRACTED"
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
        "BLOCK 6.8 FINAL ILLUMINATION ROUTE = BLOCKED"
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
        "scientific source modified   = NO"
    )

    print(
        "development metrics computed = NO"
    )

    print(
        "numeric bounds selected      = NO"
    )

    print(
        "formal content opened        = NO"
    )

    print(
        "formal outcomes read         = NO"
    )

    print(
        "policy tuning                = NO"
    )

    print()
    print(
        "Do not modify class_aware_policy.py."
    )

    print(
        "Do not run development evaluation."
    )

    print(
        "Do not run formal Stage6 evaluation."
    )

    print(
        "Send this BLOCKED section for targeted repair."
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

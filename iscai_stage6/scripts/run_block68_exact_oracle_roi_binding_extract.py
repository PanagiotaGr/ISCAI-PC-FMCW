from __future__ import annotations

from hashlib import sha256
import importlib
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

SRC = (
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

DISCOVERY = (
    S6
    / "reports/"
      "block68_oracle_roi_route_discovery.json"
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

REPORT = (
    S6
    / "reports/"
      "block68_exact_oracle_roi_binding_extract.json"
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

EXPECTED_TESTS = 202

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


def nonformal(
    path: Path,
) -> bool:

    return (
        "formal"
        not in
        str(
            path.resolve()
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


def signature_of(
    obj,
):

    try:
        return str(
            inspect.signature(
                obj
            )
        )

    except Exception:
        return "<unavailable>"


def source_lines(
    obj,
    *,
    max_lines=160,
):

    try:

        source = inspect.getsource(
            obj
        )

    except Exception as exc:

        return [
            (
                "<source unavailable: "
                f"{type(exc).__name__}: {exc}>"
            )
        ]

    return source.splitlines()[
        :max_lines
    ]


def print_source(
    label,
    obj,
):

    print()
    print(
        "------------------------------------------------------------"
    )
    print(
        label
    )
    print(
        "signature =",
        signature_of(
            obj
        ),
    )
    print(
        "------------------------------------------------------------"
    )

    lines = source_lines(
        obj
    )

    for line in lines:
        print(
            line
        )

    return {
        "signature":
            signature_of(
                obj
            ),

        "source_excerpt":
            lines,
    }


def text_occurrences(
    path: Path,
    tokens,
    *,
    context=3,
):

    source = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    lines = source.splitlines()

    results = []

    for index, line in enumerate(
        lines,
        start=1,
    ):

        lower = line.lower()

        matched = [
            token
            for token in tokens
            if token in lower
        ]

        if not matched:
            continue

        start = max(
            1,
            index - context,
        )

        end = min(
            len(
                lines
            ),
            index + context,
        )

        excerpt = []

        for line_number in range(
            start,
            end + 1,
        ):

            text = lines[
                line_number - 1
            ].rstrip()

            if len(
                text
            ) > 220:

                text = (
                    text[:217]
                    +
                    "..."
                )

            excerpt.append({
                "line":
                    line_number,

                "text":
                    text,
            })

        results.append({
            "line":
                index,

            "matched_tokens":
                matched,

            "excerpt":
                excerpt,
        })

    return results


def schema_key_paths(
    value,
    *,
    prefix="",
    max_depth=6,
    depth=0,
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

            results.append(
                path
            )

            results.extend(
                schema_key_paths(
                    child,
                    prefix=path,
                    max_depth=max_depth,
                    depth=depth + 1,
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

        results.append(
            path
        )

        results.extend(
            schema_key_paths(
                value[0],
                prefix=path,
                max_depth=max_depth,
                depth=depth + 1,
            )
        )

    return results


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
        "EXACT ORACLE / FULL-BOX / ROAD-ROI BINDING EXTRACTION"
    )
    print(
        "READ-ONLY — NO DEVELOPMENT METRICS"
    )
    print(
        "============================================================"
    )

    required = (
        BINDER,
        DISCOVERY,
        RECON,
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
            "Missing binding-extraction dependency: "
            +
            ", ".join(
                missing
            )
        ),
    )

    # ========================================================
    # A. Frozen evidence
    # ========================================================

    print()
    print(
        "===== A. FROZEN EVIDENCE ====="
    )

    require(
        sha256_file(
            BINDER
        )
        ==
        EXPECTED_BINDER_SHA,
        (
            "I_final binder SHA changed."
        ),
    )

    binder = guarded_json(
        BINDER
    )

    discovery = guarded_json(
        DISCOVERY
    )

    require(
        binder.get(
            "status"
        )
        ==
        "PASS_FINAL_ACTUATION_OUTPUT_BOUND",
        (
            "I_final binder is not PASS."
        ),
    )

    require(
        discovery.get(
            "status"
        )
        ==
        "PASS_ORACLE_ROI_ROUTE_DISCOVERY",
        (
            "Oracle/ROI discovery is not PASS."
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
        "oracle/ROI broad discovery = PASS"
    )

    # ========================================================
    # B. Exact scientific API symbols
    # ========================================================

    print()
    print(
        "===== B. EXACT FULL-BOX / OCCUPANCY APIs ====="
    )

    dp = importlib.import_module(
        "iscai_stage6.adb.deterministic_predictive"
    )

    pf = importlib.import_module(
        "iscai_stage6.adb.probabilistic_full_box"
    )

    po = importlib.import_module(
        "iscai_stage6.adb.probabilistic_occupancy"
    )

    pa = importlib.import_module(
        "iscai_stage6.adb.part_a_reactive"
    )

    exact_symbols = {
        "build_deterministic_future_full_boxes":
            getattr(
                dp,
                "build_deterministic_future_full_boxes",
                None,
            ),

        "StochasticFullBoxForecast":
            getattr(
                pf,
                "StochasticFullBoxForecast",
                None,
            ),

        "estimate_actor_occupancy_probability":
            getattr(
                po,
                "estimate_actor_occupancy_probability",
                None,
            ),
    }

    for name, obj in (
        exact_symbols.items()
    ):

        require(
            obj is not None,
            (
                f"Expected exact symbol missing: "
                f"{name}"
            ),
        )

        print()
        print(
            name
        )

        print(
            "  object type =",
            type(
                obj
            ).__name__,
        )

        print(
            "  signature =",
            signature_of(
                obj
            ),
        )

    exact_source = {}

    for name, obj in (
        exact_symbols.items()
    ):

        exact_source[
            name
        ] = print_source(
            (
                "EXACT SOURCE: "
                +
                name
            ),
            obj,
        )

    # ========================================================
    # C. Module-level oracle/road/project/raster candidates
    # ========================================================

    print()
    print(
        "===== C. MODULE-LEVEL EXACT CANDIDATES ====="
    )

    modules = {
        "deterministic_predictive":
            dp,

        "probabilistic_full_box":
            pf,

        "probabilistic_occupancy":
            po,

        "part_a_reactive":
            pa,
    }

    name_tokens = (
        "oracle",
        "future",
        "truth",
        "box",
        "project",
        "raster",
        "road",
        "roi",
        "region",
        "grid",
    )

    module_inventory = {}

    for module_name, module in (
        modules.items()
    ):

        members = []

        for name, obj in (
            inspect.getmembers(
                module
            )
        ):

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

            if (
                getattr(
                    obj,
                    "__module__",
                    None,
                )
                !=
                module.__name__
            ):
                continue

            if not any(
                token in name.lower()
                for token in name_tokens
            ):
                continue

            members.append({
                "name":
                    name,

                "kind":
                    (
                        "function"
                        if inspect.isfunction(
                            obj
                        )
                        else "class"
                    ),

                "signature":
                    signature_of(
                        obj
                    ),
            })

        module_inventory[
            module_name
        ] = members

        print()
        print(
            "MODULE =",
            module_name,
        )

        for item in members:

            print(
                f"  {item['kind']:8s} "
                f"{item['name']} "
                f"{item['signature']}"
            )

    # ========================================================
    # D. Exact road-ROI source occurrences
    # ========================================================

    print()
    print(
        "===== D. EXACT ROAD-ROI SOURCE OCCURRENCES ====="
    )

    road_tokens = (
        "road_roi",
        "road roi",
        "road_region",
        "road region",
    )

    road_occurrences = []

    source_roots = (
        SRC,
        SCRIPTS,
    )

    for root in source_roots:

        for path in sorted(
            root.glob(
                "*.py"
            )
        ):

            if not nonformal(
                path
            ):
                continue

            # Ignore this diagnostic itself.
            if (
                path.name
                ==
                Path(
                    __file__
                ).name
            ):
                continue

            occurrences = text_occurrences(
                path,
                road_tokens,
                context=4,
            )

            if not occurrences:
                continue

            relative = str(
                path.relative_to(
                    S6
                )
            )

            entry = {
                "path":
                    relative,

                "sha256":
                    sha256_file(
                        path
                    ),

                "occurrences":
                    occurrences,
            }

            road_occurrences.append(
                entry
            )

            print()
            print(
                "FILE =",
                relative,
            )

            for occurrence in occurrences:

                print(
                    "  occurrence @L",
                    occurrence[
                        "line"
                    ],
                    "tokens=",
                    occurrence[
                        "matched_tokens"
                    ],
                )

                for line in (
                    occurrence[
                        "excerpt"
                    ]
                ):

                    print(
                        f"    L{line['line']:04d}: "
                        f"{line['text']}"
                    )

    print()
    print(
        "road-ROI source files =",
        len(
            road_occurrences
        ),
    )

    # ========================================================
    # E. Exact oracle/future-truth source occurrences
    # ========================================================

    print()
    print(
        "===== E. ORACLE / FUTURE-TRUTH SOURCE OCCURRENCES ====="
    )

    oracle_tokens = (
        "oracle_mask",
        "oracle_future",
        "constructed_oracle",
        "constructed future",
        "future_gt",
        "future truth",
        "future_full_box",
        "future full-box",
    )

    oracle_occurrences = []

    for root in source_roots:

        for path in sorted(
            root.glob(
                "*.py"
            )
        ):

            if not nonformal(
                path
            ):
                continue

            if (
                path.name
                ==
                Path(
                    __file__
                ).name
            ):
                continue

            occurrences = text_occurrences(
                path,
                oracle_tokens,
                context=4,
            )

            if not occurrences:
                continue

            relative = str(
                path.relative_to(
                    S6
                )
            )

            entry = {
                "path":
                    relative,

                "sha256":
                    sha256_file(
                        path
                    ),

                "occurrences":
                    occurrences,
            }

            oracle_occurrences.append(
                entry
            )

            print()
            print(
                "FILE =",
                relative,
            )

            for occurrence in occurrences[
                :30
            ]:

                print(
                    "  occurrence @L",
                    occurrence[
                        "line"
                    ],
                    "tokens=",
                    occurrence[
                        "matched_tokens"
                    ],
                )

                for line in (
                    occurrence[
                        "excerpt"
                    ]
                ):

                    print(
                        f"    L{line['line']:04d}: "
                        f"{line['text']}"
                    )

    print()
    print(
        "oracle/future source files =",
        len(
            oracle_occurrences
        ),
    )

    # ========================================================
    # F. Development evaluator-artifact schemas
    # ========================================================

    print()
    print(
        "===== F. DEVELOPMENT EVALUATOR ARTIFACT SCHEMAS ====="
    )

    schema_tokens = (
        "oracle",
        "future",
        "full_box",
        "footprint",
        "road",
        "roi",
        "vehicle_surrogate",
        "pedestrian",
        "cyclist",
    )

    schema_evidence = {}

    candidate_paths = []

    for path in BLOCK66.rglob(
        "*"
    ):

        if not path.is_file():
            continue

        if not nonformal(
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

        lower = str(
            path
        ).lower()

        if not any(
            token in lower
            for token in schema_tokens
        ):
            continue

        candidate_paths.append(
            path
        )

    for path in sorted(
        candidate_paths
    )[
        :60
    ]:

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

        except Exception:
            continue

        keys = schema_key_paths(
            payload
        )

        relevant = [
            key
            for key in keys
            if any(
                token in key.lower()
                for token in schema_tokens
            )
        ]

        if not relevant:
            continue

        relative = str(
            path.relative_to(
                S6
            )
        )

        schema_evidence[
            relative
        ] = {
            "sha256":
                sha256_file(
                    path
                ),

            "relevant_keys":
                relevant[
                    :160
                ],
        }

        print()
        print(
            "FILE =",
            relative,
        )

        for key in relevant[
            :160
        ]:

            print(
                "  ",
                key,
            )

    print()
    print(
        "evaluator schema files =",
        len(
            schema_evidence
        ),
    )

    # ========================================================
    # G. Exact binding readiness classification
    # ========================================================

    print()
    print(
        "===== G. EXACT BINDING READINESS ====="
    )

    road_text = json.dumps(
        road_occurrences,
        sort_keys=True,
    ).lower()

    oracle_text = json.dumps(
        oracle_occurrences,
        sort_keys=True,
    ).lower()

    schema_text = json.dumps(
        schema_evidence,
        sort_keys=True,
    ).lower()

    inventory_text = json.dumps(
        module_inventory,
        sort_keys=True,
    ).lower()

    exact_readiness = {
        "deterministic_future_full_box_API":
            (
                exact_symbols[
                    "build_deterministic_future_full_boxes"
                ]
                is not None
            ),

        "stochastic_full_box_API":
            (
                exact_symbols[
                    "StochasticFullBoxForecast"
                ]
                is not None
            ),

        "occupancy_raster_API":
            (
                exact_symbols[
                    "estimate_actor_occupancy_probability"
                ]
                is not None
            ),

        "oracle_future_truth_source":
            (
                "future_gt"
                in oracle_text
                or
                "future truth"
                in oracle_text
                or
                "oracle_future"
                in oracle_text
            ),

        "oracle_mask_source":
            (
                "oracle_mask"
                in oracle_text
                or
                "oracle_mask"
                in schema_text
            ),

        "pedestrian_future_full_box_evidence":
            (
                "pedestrian"
                in
                (
                    oracle_text
                    +
                    schema_text
                    +
                    inventory_text
                )
                and
                (
                    "full_box"
                    in
                    (
                        oracle_text
                        +
                        schema_text
                        +
                        inventory_text
                    )
                    or
                    "full-box"
                    in
                    (
                        oracle_text
                        +
                        schema_text
                        +
                        inventory_text
                    )
                )
            ),

        "cyclist_future_full_box_evidence":
            (
                "cyclist"
                in
                (
                    oracle_text
                    +
                    schema_text
                    +
                    inventory_text
                )
                and
                (
                    "full_box"
                    in
                    (
                        oracle_text
                        +
                        schema_text
                        +
                        inventory_text
                    )
                    or
                    "full-box"
                    in
                    (
                        oracle_text
                        +
                        schema_text
                        +
                        inventory_text
                    )
                )
            ),

        "vehicle_surrogate_evidence":
            (
                "vehicle_surrogate"
                in schema_text
            ),

        "road_roi_code_evidence":
            (
                len(
                    road_occurrences
                )
                >
                0
            ),

        "road_roi_artifact_evidence":
            (
                "road_roi"
                in schema_text
                or
                "road roi"
                in schema_text
            ),
    }

    for key, value in (
        exact_readiness.items()
    ):

        print(
            f"{key:40s}=",
            (
                "PROVEN"
                if value
                else "NOT PROVEN"
            ),
        )

    # We require the exact already-known scientific kernels.
    require(
        exact_readiness[
            "deterministic_future_full_box_API"
        ],
        (
            "Deterministic future full-box "
            "API missing."
        ),
    )

    require(
        exact_readiness[
            "stochastic_full_box_API"
        ],
        (
            "Stochastic full-box API missing."
        ),
    )

    require(
        exact_readiness[
            "occupancy_raster_API"
        ],
        (
            "Occupancy/raster API missing."
        ),
    )

    # ========================================================
    # H. Outcome boundary
    # ========================================================

    print()
    print(
        "===== H. SCIENTIFIC OUTCOME BOUNDARY ====="
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
        "controller/policy modified          = NO"
    )

    print(
        "formal content opened               = NO"
    )

    print(
        "formal outcomes read                = NO"
    )

    print(
        "formal evaluation                   = NO"
    )

    # ========================================================
    # I. Regression
    # ========================================================

    print()
    print(
        "===== I. FULL STAGE6 REGRESSION ====="
    )

    rc, tests, output = (
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
        tests
        ==
        EXPECTED_TESTS,
        (
            "Unexpected Stage6 test count "
            f"{tests}; expected "
            f"{EXPECTED_TESTS}."
        ),
    )

    print(
        "Stage6 regression =",
        f"{tests} / {tests} PASS",
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
            "Protocol changed."
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
        "scientific runtime = UNCHANGED"
    )

    print(
        "metric authority   = UNCHANGED"
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
    # L. Extraction report
    # ========================================================

    result = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.8",

        "part":
            "2/2_exact_oracle_roi_binding_extract",

        "status":
            "PASS_EXACT_EVALUATOR_REFERENCE_SOURCE_EXTRACTED",

        "exact_symbols":
            {
                name: {
                    "object_type":
                        type(
                            obj
                        ).__name__,

                    "signature":
                        signature_of(
                            obj
                        ),
                }
                for name, obj in (
                    exact_symbols.items()
                )
            },

        "exact_source":
            exact_source,

        "module_inventory":
            module_inventory,

        "road_roi_occurrences":
            road_occurrences,

        "oracle_future_occurrences":
            oracle_occurrences,

        "development_schema_evidence":
            schema_evidence,

        "readiness":
            exact_readiness,

        "scientific_execution": {
            "development_schema_read":
                True,

            "development_metrics_computed":
                False,

            "numeric_bounds_selected":
                False,

            "policy_tuning":
                False,

            "parameter_tuning":
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
                tests,

            "status":
                "PASS",
        },

        "storage": {
            "free_GiB":
                free_gib,

            "reserve_GiB":
                MIN_FREE_GIB,

            "pass":
                True,
        },

        "next":
            (
                "Freeze exact evaluator-only oracle/full-box/"
                "road-ROI contract from extracted evidence; "
                "then execute development-only metrics and "
                "numeric non-inferiority bound freeze."
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
        "BLOCK 6.8 EXACT EVALUATOR REFERENCE EXTRACTION — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "deterministic full-box API = PROVEN"
    )

    print(
        "stochastic full-box API    = PROVEN"
    )

    print(
        "occupancy/raster API       = PROVEN"
    )

    print(
        "oracle future-truth source =",
        (
            "PROVEN"
            if exact_readiness[
                "oracle_future_truth_source"
            ]
            else "NOT PROVEN"
        ),
    )

    print(
        "oracle-mask source         =",
        (
            "PROVEN"
            if exact_readiness[
                "oracle_mask_source"
            ]
            else "NOT PROVEN"
        ),
    )

    print(
        "ped future full-box        =",
        (
            "PROVEN"
            if exact_readiness[
                "pedestrian_future_full_box_evidence"
            ]
            else "NOT PROVEN"
        ),
    )

    print(
        "cyclist future full-box    =",
        (
            "PROVEN"
            if exact_readiness[
                "cyclist_future_full_box_evidence"
            ]
            else "NOT PROVEN"
        ),
    )

    print(
        "vehicle surrogate          =",
        (
            "PROVEN"
            if exact_readiness[
                "vehicle_surrogate_evidence"
            ]
            else "NOT PROVEN"
        ),
    )

    print(
        "road ROI code evidence     =",
        (
            "PROVEN"
            if exact_readiness[
                "road_roi_code_evidence"
            ]
            else "NOT PROVEN"
        ),
    )

    print(
        "road ROI artifact evidence =",
        (
            "PROVEN"
            if exact_readiness[
                "road_roi_artifact_evidence"
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
        "Stage6 regression          =",
        f"{tests} / {tests} PASS",
    )

    print(
        "STATUS = "
        "PASS_EXACT_EVALUATOR_REFERENCE_SOURCE_EXTRACTED"
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
        "BLOCK 6.8 EXACT EVALUATOR REFERENCE EXTRACTION = BLOCKED"
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
        "scientific runtime modified  = NO"
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

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

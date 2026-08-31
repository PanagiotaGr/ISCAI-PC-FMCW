from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import traceback


ROOT = Path(
    "/home/agni/waymo"
)

STAGE3 = (
    ROOT
    / "iscai_stage3"
)

STAGE4 = (
    ROOT
    / "iscai_stage4"
)

STAGE5 = (
    ROOT
    / "iscai_stage5"
)

BLOCK57 = (
    STAGE5
    / "reports/"
      "block57_latency_controller.json"
)

LATENCY_POLICY = (
    STAGE5
    / "configs/"
      "beam_latency_policy.json"
)

BLOCK51 = (
    STAGE5
    / "reports/"
      "block51_receiver_geometry.json"
)

BLOCK52 = (
    STAGE5
    / "reports/"
      "block52_receiver_angular_posterior.json"
)

BLOCK48_SCRIPT = (
    STAGE4
    / "scripts/"
      "run_block48_formal_evaluation.py"
)

KNOWN_STAGE4_FORMAL_REPORT = (
    STAGE4
    / "reports/"
      "block48_formal_evaluation.json"
)

STAGE4_CLOSURE = (
    STAGE4
    / "reports/"
      "stage4_final_closure.json"
)

STAGE4_HANDOFF = (
    STAGE4
    / "artifacts/block410/"
      "stage4_to_stage5_handoff.json"
)

REPORT = (
    STAGE5
    / "artifacts/block58/"
      "formal_input_route_audit.json"
)


EXPECTED_BLOCK57_IMPLEMENTATION_SHA = (
    "ee5afade5b566f030bb286e030b6a6b6"
    "85113410651cccf6d294f67760255ee9"
)

EXPECTED_BLOCK57_POLICY_SHA = (
    "aa04f2e73da591b87f2ab2e962d1b69c"
    "1bad645c043c0d15970dbf0990080ed0"
)


INTERESTING_TOKENS = (
    "formal",
    "scenario",
    "prediction",
    "posterior",
    "gaussian",
    "mean",
    "covariance",
    "eligible",
    "matched",
    "receiver",
    "tracks_to_predict",
    "population",
    "sample",
    "block48",
    "120",
)

SOURCE_TOKENS = (
    "formal",
    "scenario",
    "prediction",
    "posterior",
    "gaussian",
    "mean",
    "covariance",
    "tracks_to_predict",
    "np.save",
    "np.savez",
    "artifact",
    "120",
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


def load_json(
    path: Path,
):
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
            default=str,
        )
        +
        "\n",
        encoding="utf-8",
    )

    temporary.replace(
        path
    )


def flatten_interesting(
    value,
    *,
    prefix="",
    output=None,
):
    if output is None:
        output = []

    if isinstance(
        value,
        dict,
    ):

        for key, child in (
            value.items()
        ):

            child_prefix = (
                f"{prefix}.{key}"
                if prefix
                else str(
                    key
                )
            )

            flatten_interesting(
                child,
                prefix=(
                    child_prefix
                ),
                output=(
                    output
                ),
            )

        return output

    if isinstance(
        value,
        list,
    ):

        #
        # Avoid dumping large arrays/lists.
        #
        if len(
            value
        ) > 25:

            text = (
                f"<list length={len(value)}>"
            )

            haystack = (
                prefix.lower()
                +
                " "
                +
                text.lower()
            )

            if any(
                token
                in
                haystack
                for token in (
                    INTERESTING_TOKENS
                )
            ):
                output.append(
                    {
                        "path":
                            prefix,

                        "value":
                            text,
                    }
                )

            return output

        for index, child in enumerate(
            value
        ):

            flatten_interesting(
                child,
                prefix=(
                    f"{prefix}[{index}]"
                ),
                output=(
                    output
                ),
            )

        return output

    text = str(
        value
    )

    haystack = (
        prefix.lower()
        +
        " "
        +
        text.lower()
    )

    if any(
        token
        in
        haystack
        for token in (
            INTERESTING_TOKENS
        )
    ):

        if len(
            text
        ) > 500:

            text = (
                text[
                    :500
                ]
                +
                "...<TRUNCATED>"
            )

        output.append(
            {
                "path":
                    prefix,

                "value":
                    text,
            }
        )

    return output


def summarize_json(
    path: Path,
):
    if not path.is_file():

        return {
            "path":
                str(
                    path
                ),

            "exists":
                False,
        }

    try:
        payload = load_json(
            path
        )

    except Exception as exc:

        return {
            "path":
                str(
                    path
                ),

            "exists":
                True,

            "read_error":
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
        }

    return {
        "path":
            str(
                path
            ),

        "exists":
            True,

        "sha256":
            file_sha256(
                path
            ),

        "interesting":
            flatten_interesting(
                payload
            )[
                :120
            ],
    }


def candidate_files():
    roots = (
        STAGE3
        / "reports",

        STAGE3
        / "artifacts",

        STAGE4
        / "reports",

        STAGE4
        / "artifacts",
    )

    name_tokens = (
        "formal",
        "block48",
        "scenario",
        "population",
        "prediction",
        "posterior",
        "gaussian",
        "closure",
        "handoff",
    )

    allowed_suffixes = {
        ".json",
        ".npz",
        ".npy",
        ".pkl",
        ".pickle",
        ".parquet",
        ".csv",
        ".txt",
    }

    candidates = []

    for root in roots:

        if not root.exists():
            continue

        for path in root.rglob(
            "*"
        ):

            if not path.is_file():
                continue

            if "__pycache__" in path.parts:
                continue

            if (
                path.suffix.lower()
                not in
                allowed_suffixes
            ):
                continue

            lower = (
                path.name.lower()
            )

            if not any(
                token in lower
                for token in (
                    name_tokens
                )
            ):
                continue

            stat = path.stat()

            item = {
                "path":
                    str(
                        path
                    ),

                "size_bytes":
                    int(
                        stat.st_size
                    ),

                "suffix":
                    path.suffix.lower(),
            }

            #
            # Hash reasonably sized candidate artifacts only.
            #
            if (
                stat.st_size
                <=
                256
                *
                1024
                *
                1024
            ):
                try:
                    item[
                        "sha256"
                    ] = file_sha256(
                        path
                    )

                except Exception as exc:
                    item[
                        "sha256_error"
                    ] = (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

            candidates.append(
                item
            )

    candidates.sort(
        key=lambda item:
            (
                0
                if "formal"
                in
                Path(
                    item[
                        "path"
                    ]
                ).name.lower()
                else
                1,

                item[
                    "path"
                ],
            )
    )

    return candidates[
        :120
    ]


def source_windows(
    path: Path,
):
    if not path.is_file():

        return []

    lines = path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines()

    hit_lines = []

    for index, line in enumerate(
        lines
    ):

        lower = line.lower()

        if any(
            token
            in
            lower
            for token in (
                SOURCE_TOKENS
            )
        ):
            hit_lines.append(
                index
            )

    windows = []

    for index in hit_lines:

        start = max(
            0,
            index - 3,
        )

        end = min(
            len(
                lines
            ),
            index + 4,
        )

        if (
            windows
            and
            start
            <=
            windows[
                -1
            ][
                1
            ]
        ):
            windows[
                -1
            ] = (
                windows[
                    -1
                ][
                    0
                ],
                max(
                    windows[
                        -1
                    ][
                        1
                    ],
                    end,
                ),
            )

        else:
            windows.append(
                (
                    start,
                    end,
                )
            )

    result = []

    for start, end in (
        windows[
            :30
        ]
    ):

        text = "\n".join(
            f"{number + 1:04d}: "
            f"{lines[number]}"
            for number in range(
                start,
                end,
            )
        )

        result.append(
            {
                "start_line":
                    start + 1,

                "end_line":
                    end,

                "text":
                    text,
            }
        )

    return result


def print_json_summary(
    title,
    summary,
):
    print()
    print(
        "============================================================"
    )
    print(
        title
    )
    print(
        "============================================================"
    )

    print(
        "path =",
        summary.get(
            "path"
        ),
    )

    print(
        "exists =",
        summary.get(
            "exists"
        ),
    )

    if summary.get(
        "sha256"
    ):

        print(
            "SHA256 =",
            summary[
                "sha256"
            ],
        )

    if summary.get(
        "read_error"
    ):

        print(
            "read error =",
            summary[
                "read_error"
            ],
        )

    for item in summary.get(
        "interesting",
        []
    ):

        print(
            item[
                "path"
            ],
            "=",
            item[
                "value"
            ],
        )


def main():
    print(
        "============================================================"
    )
    print(
        "STAGE 5 — BLOCK 5.8 PART 1/2"
    )
    print(
        "FROZEN FORMAL INPUT-ROUTE AUDIT"
    )
    print(
        "============================================================"
    )

    print(
        "formal N=120 read        = YES — AUDIT ONLY"
    )

    print(
        "formal metrics computed  = NO"
    )

    print(
        "Stage4 inference         = NO"
    )

    print(
        "training/recalibration   = NO"
    )

    print(
        "parameter tuning         = NO"
    )

    # ========================================================
    # A. Block5.7 continuity
    # ========================================================

    require(
        BLOCK57.is_file(),
        (
            "Block5.7 final report missing."
        ),
    )

    require(
        LATENCY_POLICY.is_file(),
        (
            "Block5.7 latency policy missing."
        ),
    )

    block57 = load_json(
        BLOCK57
    )

    require(
        block57.get(
            "completion"
        )
        ==
        "COMPLETE_FROZEN",
        (
            "Block5.7 is not COMPLETE_FROZEN."
        ),
    )

    require(
        block57[
            "implementation"
        ][
            "sha256"
        ]
        ==
        EXPECTED_BLOCK57_IMPLEMENTATION_SHA,
        (
            "Historical Block5.7 "
            "implementation SHA changed."
        ),
    )

    require(
        file_sha256(
            LATENCY_POLICY
        )
        ==
        EXPECTED_BLOCK57_POLICY_SHA,
        (
            "Frozen Block5.7 latency "
            "policy SHA changed."
        ),
    )

    print()
    print(
        "===== A. BLOCK5.7 FROZEN CONTINUITY ====="
    )

    print(
        "Block5.7              = COMPLETE / FROZEN"
    )

    print(
        "implementation SHA    = PASS"
    )

    print(
        "latency policy SHA    = PASS"
    )

    # ========================================================
    # B. Frozen Stage4 formal evidence
    # ========================================================

    summaries = {
        "Stage4 formal report":
            summarize_json(
                KNOWN_STAGE4_FORMAL_REPORT
            ),

        "Stage4 final closure":
            summarize_json(
                STAGE4_CLOSURE
            ),

        "Stage4 -> Stage5 handoff":
            summarize_json(
                STAGE4_HANDOFF
            ),

        "Stage5 receiver report":
            summarize_json(
                BLOCK51
            ),

        "Stage5 angular report":
            summarize_json(
                BLOCK52
            ),
    }

    for title, summary in (
        summaries.items()
    ):

        print_json_summary(
            title,
            summary,
        )

    # ========================================================
    # C. Candidate formal artifacts
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "C. CANDIDATE FROZEN FORMAL ARTIFACTS"
    )
    print(
        "============================================================"
    )

    candidates = candidate_files()

    print(
        "candidate count =",
        len(
            candidates
        ),
    )

    for index, item in enumerate(
        candidates,
        start=1,
    ):

        print()
        print(
            f"[{index}]"
        )

        print(
            "path =",
            item[
                "path"
            ],
        )

        print(
            "size_bytes =",
            item[
                "size_bytes"
            ],
        )

        if item.get(
            "sha256"
        ):

            print(
                "SHA256 =",
                item[
                    "sha256"
                ],
            )

    # ========================================================
    # D. Exact Block4.8 formal route source excerpts
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "D. BLOCK4.8 FORMAL EVALUATOR SOURCE WINDOWS"
    )
    print(
        "============================================================"
    )

    print(
        "path =",
        BLOCK48_SCRIPT,
    )

    windows = source_windows(
        BLOCK48_SCRIPT
    )

    print(
        "window count =",
        len(
            windows
        ),
    )

    for index, window in enumerate(
        windows,
        start=1,
    ):

        print()
        print(
            f"[SOURCE {index}] "
            f"lines "
            f"{window['start_line']}-"
            f"{window['end_line']}"
        )

        print(
            window[
                "text"
            ]
        )

    # ========================================================
    # E. Formal readiness classification
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "E. BLOCK5.8 FORMAL READINESS CLASSIFICATION"
    )
    print(
        "============================================================"
    )

    formal_report_exists = (
        KNOWN_STAGE4_FORMAL_REPORT.is_file()
    )

    closure_exists = (
        STAGE4_CLOSURE.is_file()
    )

    handoff_exists = (
        STAGE4_HANDOFF.is_file()
    )

    receiver_exists = (
        BLOCK51.is_file()
    )

    angular_exists = (
        BLOCK52.is_file()
    )

    require(
        closure_exists,
        (
            "Stage4 final closure missing."
        ),
    )

    require(
        handoff_exists,
        (
            "Stage4->Stage5 handoff missing."
        ),
    )

    require(
        receiver_exists,
        (
            "Frozen Block5.1 receiver "
            "report missing."
        ),
    )

    require(
        angular_exists,
        (
            "Frozen Block5.2 angular "
            "report missing."
        ),
    )

    #
    # Do NOT guess whether sample-level formal posterior is
    # reusable. That is determined from the printed evidence.
    #
    materialized_formal_candidates = [
        item
        for item in candidates
        if (
            "formal"
            in
            Path(
                item[
                    "path"
                ]
            ).name.lower()
            and
            Path(
                item[
                    "path"
                ]
            ).suffix.lower()
            in
            (
                ".npz",
                ".npy",
                ".pkl",
                ".pickle",
                ".parquet",
            )
        )
    ]

    print(
        "Stage4 formal report exists      =",
        formal_report_exists,
    )

    print(
        "Stage4 closure exists            =",
        closure_exists,
    )

    print(
        "Stage4->Stage5 handoff exists    =",
        handoff_exists,
    )

    print(
        "Block5.1 receiver contract       =",
        receiver_exists,
    )

    print(
        "Block5.2 angular contract        =",
        angular_exists,
    )

    print(
        "sample-level formal candidates   =",
        len(
            materialized_formal_candidates
        ),
    )

    print(
        "formal receiver selection tuning = NO"
    )

    print(
        "tracks_to_predict controller use = FORBIDDEN BY FROZEN CONTRACT"
    )

    print(
        "future truth use                 = EVALUATOR ONLY"
    )

    print(
        "Stage4 inference now             = NO"
    )

    route_status = (
        "READY_FOR_EXACT_ROUTE_SELECTION"
        if (
            formal_report_exists
            and
            closure_exists
            and
            handoff_exists
        )
        else
        "BLOCKED_MISSING_FROZEN_EVIDENCE"
    )

    payload = {
        "status":
            "PASS_READ_ONLY_AUDIT",

        "route_status":
            route_status,

        "scientific_execution": {
            "formal_N120_metadata_read":
                True,

            "formal_metrics_computed":
                False,

            "Stage4_inference":
                False,

            "training":
                False,

            "recalibration":
                False,

            "parameter_tuning":
                False,

            "upstream_modified":
                False,
        },

        "Block57_continuity": {
            "historical_implementation_sha256":
                EXPECTED_BLOCK57_IMPLEMENTATION_SHA,

            "beam_latency_policy_sha256":
                EXPECTED_BLOCK57_POLICY_SHA,
        },

        "summaries":
            summaries,

        "formal_candidates":
            candidates,

        "materialized_formal_candidates":
            materialized_formal_candidates,

        "Block48_source_windows":
            windows,

        "formal_policy": {
            "same_immutable_population_as_Stage4":
                True,

            "receiver_selection_uses_tracks_to_predict":
                False,

            "future_truth_controller_input":
                False,

            "future_truth_evaluator_only":
                True,

            "silent_receiver_dropping":
                False,

            "posthoc_tuning":
                False,
        },

        "next":
            (
                "resolve exact frozen formal "
                "scenario/posterior artifact route; "
                "then run Block5.8 Part2 formal "
                "Stage5 evaluation exactly once"
            ),
    }

    write_json(
        REPORT,
        payload,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.8 PART 1/2 FORMAL ROUTE AUDIT FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "Block5.7 continuity       = PASS"
    )

    print(
        "formal N=120 metadata     = READ-ONLY"
    )

    print(
        "formal metrics            = NOT COMPUTED"
    )

    print(
        "Stage4 inference          = NO"
    )

    print(
        "receiver policy           = FROZEN"
    )

    print(
        "angular policy            = FROZEN"
    )

    print(
        "post-hoc tuning           = NO"
    )

    print(
        "route status              =",
        route_status,
    )

    print(
        "sample-level candidates   =",
        len(
            materialized_formal_candidates
        ),
    )

    print(
        "STATUS = PASS_READ_ONLY_AUDIT"
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
        "BLOCK 5.8 PART1 FORMAL ROUTE AUDIT = BLOCKED"
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
        "formal metrics computed = NO"
    )

    print(
        "Stage4 inference = NO"
    )

    print(
        "training/recalibration = NO"
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no sys.exit().

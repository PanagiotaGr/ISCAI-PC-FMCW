from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import traceback


ROOT = Path(
    "/home/agni/waymo"
)

STAGE0 = (
    ROOT
    / "iscai_stage0"
)

STAGE1 = (
    ROOT
    / "iscai_stage1"
)

STAGE2 = (
    ROOT
    / "iscai_stage2"
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

PARTA = (
    ROOT
    / "part_a_reference/"
      "ISCAI_pc_fmcw"
)

PARTA_NOTEBOOK = (
    PARTA
    / "notebooks/"
      "ISCAI_PC_FMCW.ipynb"
)

BLOCK56 = (
    STAGE5
    / "reports/"
      "block56_optical_link.json"
)

OPTICAL_POLICY = (
    STAGE5
    / "configs/"
      "optical_link_policy.json"
)

REPORT = (
    STAGE5
    / "artifacts/block57/"
      "timing_source_audit.json"
)


EXPECTED_BLOCK56_IMPLEMENTATION_SHA = (
    "75647e018841a65760b122d619dd11f9"
    "cd5e74e551ece0f64ea19e6189486df9"
)

EXPECTED_BLOCK56_POLICY_SHA = (
    "2075f0d0f3128af94c5d5ed99f0da4d"
    "ba87e08ace69de542bec4c9dd4454302b"
)


TIMING_TERMS = (
    "tchirp",
    "t_chirp",
    "chirp duration",
    "chirp_duration",
    "10e-6",
    "1e-5",
    "10 us",
    "10 µs",
    "frame interval",
    "frame_interval",
    "frame time",
    "frame_time",
    "0.1",
    "100 ms",
    "100ms",
    "10 hz",
    "10hz",
    "sample interval",
    "sample_interval",
    "dt_s",
    "delta_t",
    "time_step",
    "timestep",
)

LATENCY_TERMS = (
    "perf_counter",
    "cuda.synchronize",
    "synchronize()",
    "elapsed",
    "latency",
    "runtime",
    "timing",
    "memory_allocated",
    "max_memory_allocated",
    "numel",
    "parameter_count",
    "flops",
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


def numeric_tokens(
    text,
):
    return re.findall(
        r"(?<![A-Za-z_])"
        r"[-+]?"
        r"(?:\d+\.\d+|\d+)"
        r"(?:[eE][-+]?\d+)?",
        text,
    )[
        :30
    ]


def notebook_cell_text(
    cell,
):
    source = cell.get(
        "source",
        []
    )

    if isinstance(
        source,
        list,
    ):
        return "".join(
            source
        )

    return str(
        source
    )


def scan_parta_timing():
    payload = json.loads(
        PARTA_NOTEBOOK.read_text(
            encoding="utf-8"
        )
    )

    hits = []

    for index, cell in enumerate(
        payload.get(
            "cells",
            []
        )
    ):
        text = notebook_cell_text(
            cell
        )

        lower = text.lower()

        matched = [
            term
            for term in TIMING_TERMS
            if term in lower
        ]

        if not matched:
            continue

        score = (
            5
            *
            len(
                matched
            )
        )

        if (
            "tchirp"
            in lower
            or
            "t_chirp"
            in lower
        ):
            score += 100

        if (
            "10e-6"
            in lower
            or
            "1e-5"
            in lower
        ):
            score += 40

        hits.append({
            "cell":
                index,

            "score":
                score,

            "matched":
                matched,

            "numbers":
                numeric_tokens(
                    text
                ),

            "text":
                text,
        })

    hits.sort(
        key=lambda item:
            (
                item[
                    "score"
                ],
                item[
                    "cell"
                ],
            ),
        reverse=True,
    )

    return hits


def scan_tree(
    root: Path,
    *,
    search_terms,
    suffixes=None,
):
    if suffixes is None:
        suffixes = {
            ".py",
            ".json",
            ".yaml",
            ".yml",
            ".md",
            ".txt",
        }

    hits = []

    files_scanned = 0

    if not root.exists():
        return (
            files_scanned,
            hits,
        )

    for path in sorted(
        root.rglob(
            "*"
        ),
        key=str,
    ):

        if not path.is_file():
            continue

        if "__pycache__" in path.parts:
            continue

        if (
            path.suffix.lower()
            not in
            suffixes
        ):
            continue

        files_scanned += 1

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

        except Exception:
            continue

        for line_number, line in enumerate(
            text.splitlines(),
            start=1,
        ):
            lower = line.lower()

            matched = [
                term
                for term in search_terms
                if term in lower
            ]

            if not matched:
                continue

            score = (
                5
                *
                len(
                    matched
                )
            )

            if any(
                token in lower
                for token in (
                    "0.1",
                    "100 ms",
                    "100ms",
                    "10 hz",
                    "10hz",
                )
            ):
                score += 20

            if any(
                token in lower
                for token in (
                    "perf_counter",
                    "cuda.synchronize",
                    "memory_allocated",
                    "max_memory_allocated",
                )
            ):
                score += 30

            hits.append({
                "path":
                    str(
                        path
                    ),

                "line":
                    line_number,

                "score":
                    score,

                "matched":
                    matched,

                "numbers":
                    numeric_tokens(
                        line
                    ),

                "text":
                    line,
            })

    hits.sort(
        key=lambda item:
            (
                item[
                    "score"
                ],
                item[
                    "path"
                ],
                item[
                    "line"
                ],
            ),
        reverse=True,
    )

    return (
        files_scanned,
        hits,
    )


def print_hits(
    title,
    hits,
    *,
    limit,
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
        "hit count =",
        len(
            hits
        ),
    )

    for rank, hit in enumerate(
        hits[
            :limit
        ],
        start=1,
    ):
        print()
        print(
            f"[{rank}] score =",
            hit[
                "score"
            ],
        )

        if "cell" in hit:
            print(
                "cell =",
                hit[
                    "cell"
                ],
            )

        if "path" in hit:
            print(
                "path =",
                hit[
                    "path"
                ],
            )

            print(
                "line =",
                hit[
                    "line"
                ],
            )

        print(
            "matched =",
            hit[
                "matched"
            ],
        )

        print(
            "numbers =",
            hit[
                "numbers"
            ],
        )

        print(
            "source ="
        )

        text = str(
            hit[
                "text"
            ]
        )

        if (
            "data:image/"
            in text
            and
            "base64,"
            in text
        ):
            print(
                "<EMBEDDED IMAGE OMITTED>"
            )

        else:
            print(
                text[
                    :5000
                ]
            )


def main():
    print(
        "============================================================"
    )
    print(
        "STAGE 5 — BLOCK 5.7 PART 1/2"
    )
    print(
        "EXACT TIMING / LATENCY SOURCE AUDIT"
    )
    print(
        "============================================================"
    )

    print(
        "read-only                = YES"
    )

    print(
        "formal N=120 used        = NO"
    )

    print(
        "Stage4 inference         = NO"
    )

    print(
        "training/recalibration   = NO"
    )

    print(
        "Tbeam frozen here        = NO"
    )

    print(
        "Tframe frozen here       = NO"
    )

    # ========================================================
    # A. Block5.6 continuity
    # ========================================================

    require(
        BLOCK56.is_file(),
        (
            "Block5.6 report missing."
        ),
    )

    require(
        OPTICAL_POLICY.is_file(),
        (
            "Block5.6 optical policy missing."
        ),
    )

    block56 = json.loads(
        BLOCK56.read_text(
            encoding="utf-8"
        )
    )

    require(
        block56.get(
            "completion"
        )
        ==
        "COMPLETE_FROZEN",
        (
            "Block5.6 is not COMPLETE_FROZEN."
        ),
    )

    require(
        block56[
            "implementation"
        ][
            "sha256"
        ]
        ==
        EXPECTED_BLOCK56_IMPLEMENTATION_SHA,
        (
            "Historical Block5.6 "
            "implementation SHA changed."
        ),
    )

    require(
        file_sha256(
            OPTICAL_POLICY
        )
        ==
        EXPECTED_BLOCK56_POLICY_SHA,
        (
            "Frozen optical policy "
            "SHA changed."
        ),
    )

    print()
    print(
        "===== A. BLOCK5.6 CONTINUITY ====="
    )

    print(
        "Block5.6                = COMPLETE / FROZEN"
    )

    print(
        "implementation SHA      = PASS"
    )

    print(
        "optical policy SHA      = PASS"
    )

    # ========================================================
    # B. Part-A beam-probe timing evidence
    # ========================================================

    require(
        PARTA_NOTEBOOK.is_file(),
        (
            "Frozen Part-A notebook missing."
        ),
    )

    parta_hits = (
        scan_parta_timing()
    )

    print_hits(
        "B. PART-A TIMING EVIDENCE",
        parta_hits,
        limit=12,
    )

    # ========================================================
    # C. WOMD/control-frame timing evidence
    # ========================================================

    timing_hits = []

    timing_files = 0

    for root in (
        STAGE0,
        STAGE1,
        STAGE2,
        STAGE3,
        STAGE4,
    ):

        files_scanned, hits = scan_tree(
            root,
            search_terms=(
                TIMING_TERMS
            ),
        )

        timing_files += (
            files_scanned
        )

        timing_hits.extend(
            hits
        )

    timing_hits.sort(
        key=lambda item:
            (
                item[
                    "score"
                ],
                item[
                    "path"
                ],
                item[
                    "line"
                ],
            ),
        reverse=True,
    )

    print_hits(
        "C. WOMD / CONTROL-INTERVAL EVIDENCE",
        timing_hits,
        limit=20,
    )

    # ========================================================
    # D. Existing latency instrumentation
    # ========================================================

    latency_hits = []

    latency_files = 0

    for root in (
        STAGE3,
        STAGE4,
        STAGE5,
    ):

        files_scanned, hits = scan_tree(
            root,
            search_terms=(
                LATENCY_TERMS
            ),
        )

        latency_files += (
            files_scanned
        )

        latency_hits.extend(
            hits
        )

    latency_hits.sort(
        key=lambda item:
            (
                item[
                    "score"
                ],
                item[
                    "path"
                ],
                item[
                    "line"
                ],
            ),
        reverse=True,
    )

    print_hits(
        "D. EXISTING LATENCY / RUNTIME INSTRUMENTATION",
        latency_hits,
        limit=20,
    )

    # ========================================================
    # E. Classification only
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "E. TIMING AUDIT CLASSIFICATION"
    )
    print(
        "============================================================"
    )

    tchirp_hits = [
        hit
        for hit in parta_hits
        if (
            "tchirp"
            in str(
                hit[
                    "text"
                ]
            ).lower()
            or
            "t_chirp"
            in str(
                hit[
                    "text"
                ]
            ).lower()
        )
    ]

    frame_100ms_hits = [
        hit
        for hit in timing_hits
        if any(
            token
            in
            str(
                hit[
                    "text"
                ]
            ).lower()
            for token in (
                "0.1",
                "100 ms",
                "100ms",
                "10 hz",
                "10hz",
            )
        )
    ]

    print(
        "Part-A chirp-duration hits =",
        len(
            tchirp_hits
        ),
    )

    print(
        "100-ms/control hits        =",
        len(
            frame_100ms_hits
        ),
    )

    print(
        "latency instrumentation hits =",
        len(
            latency_hits
        ),
    )

    payload = {
        "status":
            "PASS_READ_ONLY_AUDIT",

        "scientific_execution": {
            "formal_N120_read":
                False,

            "Stage4_inference":
                False,

            "training":
                False,

            "recalibration":
                False,

            "upstream_modified":
                False,
        },

        "Block56_continuity": {
            "status":
                "PASS",

            "historical_implementation_sha256":
                EXPECTED_BLOCK56_IMPLEMENTATION_SHA,

            "optical_policy_sha256":
                EXPECTED_BLOCK56_POLICY_SHA,
        },

        "PartA_timing": {
            "hits":
                parta_hits,

            "Tchirp_candidate_hits":
                tchirp_hits,

            "Tbeam_decision":
                "NOT_YET_FROZEN",
        },

        "control_frame_timing": {
            "files_scanned":
                timing_files,

            "hits":
                timing_hits,

            "100ms_candidate_hits":
                frame_100ms_hits,

            "Tframe_decision":
                "NOT_YET_FROZEN",
        },

        "runtime_instrumentation": {
            "files_scanned":
                latency_files,

            "hits":
                latency_hits,
        },

        "next":
            (
                "freeze Tbeam/Tframe and "
                "Stage5 latency-component "
                "measurement contract only "
                "after exact evidence review"
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
        "BLOCK 5.7 PART 1/2 TIMING AUDIT FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "Block5.6 continuity       = PASS"
    )

    print(
        "Part-A timing evidence    = AUDITED"
    )

    print(
        "control interval evidence = AUDITED"
    )

    print(
        "runtime instrumentation   = AUDITED"
    )

    print(
        "Tbeam                     = NOT YET FROZEN"
    )

    print(
        "Tframe                    = NOT YET FROZEN"
    )

    print(
        "formal N=120 used         = NO"
    )

    print(
        "Stage4 inference          = NO"
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
        "BLOCK 5.7 PART1 TIMING AUDIT = BLOCKED"
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
        "formal N=120 used = NO"
    )

    print(
        "Stage4 inference = NO"
    )

    print(
        "upstream modified = NO"
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no sys.exit().

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

STAGE5 = (
    ROOT
    / "iscai_stage5"
)

PARTA_ROOT = (
    ROOT
    / "part_a_reference/"
      "ISCAI_pc_fmcw"
)

NOTEBOOK = (
    PARTA_ROOT
    / "notebooks/"
      "ISCAI_PC_FMCW.ipynb"
)

PARTA_FROZEN_REFERENCE = (
    STAGE0
    / "reports/stage0/"
      "part_a_frozen_reference.json"
)

REPORT = (
    STAGE5
    / "artifacts/block56/"
      "exact_parta_optical_headlamp_audit.json"
)


EXPECTED_NOTEBOOK_SHA = (
    "b5a80a6d3441de6d571db4f65b4a43ed"
    "4052cc2b3ccba935ad31b5dd51316ef3"
)

EXPECTED_PARTA_REFERENCE_SHA = (
    "fcdfc0a7b9c14fa9447ceb8563a6f20a"
    "c277706223388dc9aa8a1edf416c5e2b"
)


OPTICAL_TERMS = (
    "dpsk",
    "ber",
    "snr",
    "received power",
    "received_power",
    "prx",
    "p_rx",
    "optical gain",
    "gopt",
    "g_opt",
    "pointing",
    "noise",
    "n0",
    "bandwidth",
    "data rate",
    "data_rate",
    "bit rate",
    "bit_rate",
    "attenuation",
    "atmospheric",
    "responsivity",
    "photodetector",
)

STRONG_OPTICAL_TERMS = (
    "dpsk",
    "ber",
    "received power",
    "received_power",
    "prx",
    "p_rx",
    "gopt",
    "g_opt",
    "snr",
)

HEADLAMP_TERMS = (
    "headlamp",
    "extrinsic",
    "t_h0",
    "h0_from",
    "from_h0",
    "front_face_midpoint",
    "transmitter",
    "tx_",
    "optical",
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
        :40
    ]


def optical_score(
    text,
):
    lower = text.lower()

    score = 0

    for term in OPTICAL_TERMS:

        if term in lower:
            score += 4

    for term in STRONG_OPTICAL_TERMS:

        if term in lower:
            score += 20

    if numeric_tokens(
        text
    ):
        score += 5

    return score


def collect_notebook_hits(
    payload,
):
    hits = []

    cells = payload.get(
        "cells",
        []
    )

    for index, cell in enumerate(
        cells
    ):
        text = notebook_cell_text(
            cell
        )

        lower = text.lower()

        matched = [
            term
            for term in OPTICAL_TERMS
            if term in lower
        ]

        if not matched:
            continue

        hits.append({
            "cell_index":
                index,

            "cell_type":
                cell.get(
                    "cell_type"
                ),

            "matched":
                matched,

            "score":
                optical_score(
                    text
                ),

            "numbers":
                numeric_tokens(
                    text
                ),

            "text":
                text,
        })

    hits.sort(
        key=lambda hit:
            (
                hit[
                    "score"
                ],
                hit[
                    "cell_index"
                ],
            ),
        reverse=True,
    )

    return hits


def exact_cells(
    payload,
    indices,
):
    cells = payload.get(
        "cells",
        []
    )

    result = []

    for index in indices:

        if (
            index
            <
            0
            or
            index
            >=
            len(
                cells
            )
        ):
            continue

        text = notebook_cell_text(
            cells[
                index
            ]
        )

        result.append({
            "cell_index":
                index,

            "cell_type":
                cells[
                    index
                ].get(
                    "cell_type"
                ),

            "numbers":
                numeric_tokens(
                    text
                ),

            "text":
                text,
        })

    return result


def effective_rate_hits(
    payload,
):
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

        if (
            "effective_rate"
            in lower
            or
            "effective rate"
            in lower
            or
            "effective throughput"
            in lower
        ):
            hits.append({
                "cell_index":
                    index,

                "text":
                    text,
            })

    return hits


def scan_stage1_headlamp():
    roots = (
        STAGE1
        / "src",

        STAGE1
        / "configs",

        STAGE1
        / "reports",
    )

    allowed_suffixes = {
        ".py",
        ".json",
        ".yaml",
        ".yml",
        ".md",
        ".txt",
    }

    hits = []

    files_scanned = 0

    for root in roots:

        if not root.exists():
            continue

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
                allowed_suffixes
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
                    for term in HEADLAMP_TERMS
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

                if "headlamp" in lower:
                    score += 20

                if "extrinsic" in lower:
                    score += 20

                if (
                    "t_h0"
                    in lower
                    or
                    "h0_from"
                    in lower
                    or
                    "from_h0"
                    in lower
                ):
                    score += 15

                if numeric_tokens(
                    line
                ):
                    score += 2

                hits.append({
                    "path":
                        str(
                            path
                        ),

                    "line":
                        line_number,

                    "matched":
                        matched,

                    "score":
                        score,

                    "text":
                        line,
                })

    hits.sort(
        key=lambda hit:
            (
                hit[
                    "score"
                ],
                hit[
                    "path"
                ],
                hit[
                    "line"
                ],
            ),
        reverse=True,
    )

    return (
        files_scanned,
        hits,
    )


def print_cell(
    title,
    item,
    *,
    max_chars=12000,
):
    print()
    print(
        "------------------------------------------------------------"
    )

    print(
        title
    )

    print(
        "------------------------------------------------------------"
    )

    print(
        "cell index =",
        item[
            "cell_index"
        ],
    )

    print(
        "numbers =",
        item.get(
            "numbers"
        ),
    )

    print(
        "source ="
    )

    print(
        item[
            "text"
        ][
            :max_chars
        ]
    )


def main():
    print(
        "============================================================"
    )

    print(
        "BLOCK 5.6 PART 1 — EXACT OPTICAL / HEADLAMP SOURCE AUDIT"
    )

    print(
        "============================================================"
    )

    print(
        "read-only                  = YES"
    )

    print(
        "formal N=120 used          = NO"
    )

    print(
        "Stage4 inference           = NO"
    )

    print(
        "training/recalibration     = NO"
    )

    print(
        "optical formula frozen now = NO"
    )

    print(
        "effective-rate frozen now  = NO"
    )

    require(
        NOTEBOOK.is_file(),
        (
            "Frozen Part-A notebook missing."
        ),
    )

    require(
        PARTA_FROZEN_REFERENCE.is_file(),
        (
            "Stage0 Part-A frozen reference missing."
        ),
    )

    notebook_sha = file_sha256(
        NOTEBOOK
    )

    reference_sha = file_sha256(
        PARTA_FROZEN_REFERENCE
    )

    require(
        notebook_sha
        ==
        EXPECTED_NOTEBOOK_SHA,
        (
            "Frozen Part-A notebook SHA changed."
        ),
    )

    require(
        reference_sha
        ==
        EXPECTED_PARTA_REFERENCE_SHA,
        (
            "Frozen Stage0 Part-A reference "
            "SHA changed."
        ),
    )

    notebook = json.loads(
        NOTEBOOK.read_text(
            encoding="utf-8"
        )
    )

    print()
    print(
        "============================================================"
    )

    print(
        "A. FROZEN PART-A INTEGRITY"
    )

    print(
        "============================================================"
    )

    print(
        "notebook SHA256 =",
        notebook_sha,
    )

    print(
        "Stage0 Part-A reference SHA256 =",
        reference_sha,
    )

    print(
        "cell count =",
        len(
            notebook.get(
                "cells",
                []
            )
        ),
    )

    print(
        "integrity = PASS"
    )

    # ========================================================
    # B. Exact late optical cells
    # ========================================================

    print()
    print(
        "============================================================"
    )

    print(
        "B. EXACT PART-A CELLS 63 / 64 / 65"
    )

    print(
        "============================================================"
    )

    late_cells = exact_cells(
        notebook,
        (
            63,
            64,
            65,
        ),
    )

    require(
        late_cells,
        (
            "Expected Part-A optical cells "
            "63/64/65 unavailable."
        ),
    )

    for item in late_cells:

        print_cell(
            (
                f"PART-A CELL "
                f"{item['cell_index']}"
            ),
            item,
        )

    # ========================================================
    # C. Ranked DPSK / BER / SNR / optical source
    # ========================================================

    print()
    print(
        "============================================================"
    )

    print(
        "C. RANKED PART-A OPTICAL / DPSK SOURCE HITS"
    )

    print(
        "============================================================"
    )

    optical_hits = (
        collect_notebook_hits(
            notebook
        )
    )

    print(
        "total optical/DPSK hits =",
        len(
            optical_hits
        ),
    )

    for rank, hit in enumerate(
        optical_hits[
            :12
        ],
        start=1,
    ):

        print()
        print(
            f"[OPTICAL {rank}]"
        )

        print(
            "cell =",
            hit[
                "cell_index"
            ],
        )

        print(
            "score =",
            hit[
                "score"
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

        print(
            hit[
                "text"
            ][
                :8000
            ]
        )

    # ========================================================
    # D. Effective-rate source status
    # ========================================================

    print()
    print(
        "============================================================"
    )

    print(
        "D. EFFECTIVE-RATE SOURCE STATUS"
    )

    print(
        "============================================================"
    )

    rate_hits = (
        effective_rate_hits(
            notebook
        )
    )

    print(
        "effective-rate source hits =",
        len(
            rate_hits
        ),
    )

    if rate_hits:

        for hit in rate_hits:

            print()
            print(
                "cell =",
                hit[
                    "cell_index"
                ],
            )

            print(
                hit[
                    "text"
                ][
                    :5000
                ]
            )

    else:

        print(
            "Part-A source-level effective-rate formula = NOT FOUND"
        )

        print(
            "Stage5 action = DERIVE FROM FROZEN RAW LINK RATE + BEAM PROBING OVERHEAD"
        )

    # ========================================================
    # E. Stage1 headlamp / H0 extrinsic evidence
    # ========================================================

    print()
    print(
        "============================================================"
    )

    print(
        "E. STAGE1 HEADLAMP / H0 EXTRINSIC EVIDENCE"
    )

    print(
        "============================================================"
    )

    (
        stage1_files,
        headlamp_hits,
    ) = scan_stage1_headlamp()

    print(
        "Stage1 files scanned =",
        stage1_files,
    )

    print(
        "headlamp/extrinsic hits =",
        len(
            headlamp_hits
        ),
    )

    for rank, hit in enumerate(
        headlamp_hits[
            :25
        ],
        start=1,
    ):

        print()
        print(
            f"[HEADLAMP {rank}]"
        )

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
            "score =",
            hit[
                "score"
            ],
        )

        print(
            "matched =",
            hit[
                "matched"
            ],
        )

        print(
            "text =",
            hit[
                "text"
            ],
        )

    # ========================================================
    # F. Audit-only decision
    # ========================================================

    print()
    print(
        "============================================================"
    )

    print(
        "F. AUDIT DECISION"
    )

    print(
        "============================================================"
    )

    strong_optical = [
        hit
        for hit in optical_hits
        if any(
            term
            in
            hit[
                "matched"
            ]
            for term in (
                "dpsk",
                "ber",
                "snr",
                "received power",
                "received_power",
                "prx",
                "p_rx",
                "gopt",
                "g_opt",
            )
        )
    ]

    require(
        strong_optical,
        (
            "No strong Part-A optical/DPSK "
            "source evidence found."
        ),
    )

    headlamp_found = bool(
        headlamp_hits
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

        "PartA_integrity": {
            "notebook":
                str(
                    NOTEBOOK
                ),

            "notebook_sha256":
                notebook_sha,

            "Stage0_reference":
                str(
                    PARTA_FROZEN_REFERENCE
                ),

            "Stage0_reference_sha256":
                reference_sha,
        },

        "PartA_exact_cells_63_64_65":
            late_cells,

        "PartA_optical_ranked_hits":
            optical_hits,

        "effective_rate": {
            "source_hit_count":
                len(
                    rate_hits
                ),

            "source_defined":
                bool(
                    rate_hits
                ),

            "Stage5_status":
                (
                    "MUST_FREEZE_DERIVED_FORMULA_"
                    "BEFORE_FORMAL"
                ),
        },

        "Stage1_headlamp_H0": {
            "files_scanned":
                stage1_files,

            "hit_count":
                len(
                    headlamp_hits
                ),

            "hits":
                headlamp_hits,

            "evidence_found":
                headlamp_found,

            "decision":
                (
                    "INSPECT_EXACT_TRANSFORM_EVIDENCE_"
                    "BEFORE_OPTICAL_IMPLEMENTATION"
                ),
        },

        "next": {
            "PartA":
                (
                    "freeze exact optical/link/DPSK "
                    "equations and constants only "
                    "from source evidence"
                ),

            "headlamp":
                (
                    "resolve exact H0-to-headlamp "
                    "transmitter frame semantics"
                ),

            "effective_rate":
                (
                    "freeze Stage5-derived beam-"
                    "probing-overhead formula "
                    "before formal evaluation"
                ),
        },
    }

    write_json(
        REPORT,
        payload,
    )

    print(
        "Part-A integrity            = PASS"
    )

    print(
        "strong optical/DPSK source  = FOUND"
    )

    print(
        "effective-rate source hits  =",
        len(
            rate_hits
        ),
    )

    print(
        "Stage1 headlamp/H0 hits     =",
        len(
            headlamp_hits
        ),
    )

    print(
        "optical implementation      = NOT STARTED"
    )

    print(
        "formal N=120 used           = NO"
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
        "BLOCK 5.6 PART1 SOURCE AUDIT = BLOCKED"
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

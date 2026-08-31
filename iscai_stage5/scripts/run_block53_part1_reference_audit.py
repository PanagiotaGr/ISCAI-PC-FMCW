from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import traceback


ROOT = Path(
    "/home/agni/waymo"
)

PARTA = (
    ROOT
    / "part_a_reference/"
      "ISCAI_pc_fmcw"
)

LEGACY = (
    ROOT
    / "iscai_stage5_panagiota"
)

STAGE5 = (
    ROOT
    / "iscai_stage5"
)

REPORT = (
    STAGE5
    / "artifacts/block53/"
      "parta_beam_reference_audit.json"
)


KEYWORDS = (
    "beam",
    "beamwidth",
    "beam width",
    "codebook",
    "azimuth",
    "theta",
    "field of view",
    "fov",
    "headlamp",
    "optical gain",
    "gain",
    "divergence",
    "half-power",
    "half power",
    "pointing",
)

STRONG_KEYWORDS = (
    "beamwidth",
    "beam width",
    "codebook",
    "field of view",
    "fov",
    "optical gain",
    "divergence",
    "half-power",
    "half power",
)

ALLOWED_SUFFIXES = {
    ".py",
    ".ipynb",
    ".json",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}


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


def numbers_in(
    text,
):
    return re.findall(
        r"[-+]?"
        r"(?:\d+\.\d+|\d+)"
        r"(?:[eE][-+]?\d+)?",
        text,
    )[
        :20
    ]


def score_hit(
    text,
    *,
    is_parta,
):
    lower = text.lower()

    score = (
        25
        if is_parta
        else
        0
    )

    for keyword in KEYWORDS:

        if keyword in lower:
            score += 2

    for keyword in STRONG_KEYWORDS:

        if keyword in lower:
            score += 8

    if numbers_in(
        text
    ):
        score += 3

    return score


def scan_text_file(
    path: Path,
    *,
    root_name,
):
    try:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    except Exception:
        return []

    hits = []

    lines = text.splitlines()

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        lower = line.lower()

        matched = [
            keyword
            for keyword in KEYWORDS
            if keyword in lower
        ]

        if not matched:
            continue

        hits.append({
            "source_root":
                root_name,

            "path":
                str(
                    path
                ),

            "location":
                f"line:{line_number}",

            "keywords":
                matched,

            "numbers":
                numbers_in(
                    line
                ),

            "text":
                line[
                    :1000
                ],

            "score":
                score_hit(
                    line,
                    is_parta=(
                        root_name
                        ==
                        "PartA"
                    ),
                ),
        })

    return hits


def scan_notebook(
    path: Path,
    *,
    root_name,
):
    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except Exception:
        return []

    hits = []

    cells = payload.get(
        "cells",
        []
    )

    for cell_index, cell in enumerate(
        cells
    ):

        source = cell.get(
            "source",
            []
        )

        if isinstance(
            source,
            list,
        ):
            text = "".join(
                source
            )

        else:
            text = str(
                source
            )

        lower = text.lower()

        matched = [
            keyword
            for keyword in KEYWORDS
            if keyword in lower
        ]

        if not matched:
            continue

        hits.append({
            "source_root":
                root_name,

            "path":
                str(
                    path
                ),

            "location":
                f"cell:{cell_index}",

            "keywords":
                matched,

            "numbers":
                numbers_in(
                    text
                ),

            "text":
                text[
                    :3000
                ],

            "score":
                score_hit(
                    text,
                    is_parta=(
                        root_name
                        ==
                        "PartA"
                    ),
                ),
        })

    return hits


def scan_root(
    root: Path,
    *,
    root_name,
):
    if not root.exists():
        return (
            0,
            [],
        )

    file_count = 0

    hits = []

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
            ALLOWED_SUFFIXES
        ):
            continue

        file_count += 1

        if (
            path.suffix.lower()
            ==
            ".ipynb"
        ):
            hits.extend(
                scan_notebook(
                    path,
                    root_name=(
                        root_name
                    ),
                )
            )

        else:
            hits.extend(
                scan_text_file(
                    path,
                    root_name=(
                        root_name
                    ),
                )
            )

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
                    "location"
                ],
            ),
        reverse=True,
    )

    return (
        file_count,
        hits,
    )


def main():
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.3 PART 1 — PART-A BEAM REFERENCE AUDIT"
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
        "beam support frozen here   = NO"
    )

    print(
        "physical gain frozen here  = NO"
    )

    (
        parta_files,
        parta_hits,
    ) = scan_root(
        PARTA,
        root_name="PartA",
    )

    (
        legacy_files,
        legacy_hits,
    ) = scan_root(
        LEGACY,
        root_name="LegacyStage5Reference",
    )

    print()
    print(
        "===== PART-A REFERENCE ====="
    )

    print(
        "files scanned =",
        parta_files,
    )

    print(
        "beam/gain hits =",
        len(
            parta_hits
        ),
    )

    for index, hit in enumerate(
        parta_hits[
            :20
        ],
        start=1,
    ):

        print()
        print(
            f"[PartA {index}]"
        )

        print(
            "path =",
            hit[
                "path"
            ],
        )

        print(
            "location =",
            hit[
                "location"
            ],
        )

        print(
            "keywords =",
            hit[
                "keywords"
            ],
        )

        print(
            "numbers =",
            hit[
                "numbers"
            ],
        )

        print(
            "text =",
            hit[
                "text"
            ][
                :1600
            ],
        )

    print()
    print(
        "===== LEGACY STAGE5 REFERENCE ====="
    )

    print(
        "files scanned =",
        legacy_files,
    )

    print(
        "beam/gain hits =",
        len(
            legacy_hits
        ),
    )

    for index, hit in enumerate(
        legacy_hits[
            :10
        ],
        start=1,
    ):

        print()
        print(
            f"[Legacy {index}]"
        )

        print(
            "path =",
            hit[
                "path"
            ],
        )

        print(
            "location =",
            hit[
                "location"
            ],
        )

        print(
            "keywords =",
            hit[
                "keywords"
            ],
        )

        print(
            "numbers =",
            hit[
                "numbers"
            ],
        )

        print(
            "text =",
            hit[
                "text"
            ][
                :1200
            ],
        )

    payload = {
        "status":
            "PASS_READ_ONLY_REFERENCE_AUDIT",

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

        "PartA": {
            "root":
                str(
                    PARTA
                ),

            "files_scanned":
                parta_files,

            "hits":
                parta_hits,
        },

        "legacy_Stage5_reference": {
            "root":
                str(
                    LEGACY
                ),

            "runtime_dependency":
                False,

            "files_scanned":
                legacy_files,

            "hits":
                legacy_hits,
        },

        "decision": {
            "beam_counts":
                [
                    16,
                    32,
                    64,
                ],

            "beam_counts_status":
                "FROZEN_FROM_STAGE5_CONTRACT",

            "angular_support":
                "NOT_YET_FROZEN",

            "physical_gain_pattern":
                "NOT_YET_FROZEN",

            "decision_cells":
                (
                    "nonoverlapping_probability_"
                    "accounting_cells"
                ),

            "physical_gain_patterns":
                "may_overlap",

            "next":
                (
                    "Block5.3_Part2_freeze_"
                    "support_and_gain_from_"
                    "PartA_or_explicit_"
                    "constructed_dev_contract"
                ),
        },
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
        "REFERENCE AUDIT FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "Part-A files scanned        =",
        parta_files,
    )

    print(
        "Part-A beam/gain hits       =",
        len(
            parta_hits
        ),
    )

    print(
        "legacy runtime dependency   = NO"
    )

    print(
        "angular support             = NOT YET FROZEN"
    )

    print(
        "physical gain pattern       = NOT YET FROZEN"
    )

    print(
        "formal N=120 used           = NO"
    )

    print(
        "STATUS = PASS"
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
        "PART-A BEAM REFERENCE AUDIT = BLOCKED"
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

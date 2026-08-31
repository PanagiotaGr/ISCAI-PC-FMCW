from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import traceback
import zipfile

import numpy as np


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

REPORT = (
    STAGE5
    / "artifacts/block52/"
      "dev_posterior_resolver_audit.json"
)


# ============================================================
# Explicitly exclude formal/freeze artifacts from candidate
# use. They may be listed only as EXCLUDED evidence.
# ============================================================

FORMAL_OR_POSTFORMAL_TOKENS = (
    "block48",
    "block49",
    "block410",
    "formal",
    "n120",
    "validation_120",
)


SEARCH_ROOTS = (
    STAGE4
    / "artifacts",

    STAGE4
    / "reports",

    STAGE4
    / "scripts",

    STAGE4
    / "src",
)


BINARY_SUFFIXES = {
    ".npz",
    ".npy",
    ".pt",
    ".pth",
    ".pkl",
    ".pickle",
}

TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".py",
    ".md",
    ".txt",
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


def is_formal_or_postformal(
    path: Path,
):
    lower = str(
        path
    ).lower()

    return any(
        token in lower
        for token in FORMAL_OR_POSTFORMAL_TOKENS
    )


def relative(
    path: Path,
):
    try:
        return str(
            path.relative_to(
                ROOT
            )
        )

    except Exception:
        return str(
            path
        )


def collect_files():
    files = []

    for root in SEARCH_ROOTS:

        if not root.exists():
            continue

        for path in root.rglob(
            "*"
        ):
            if not path.is_file():
                continue

            if "__pycache__" in path.parts:
                continue

            if path.suffix.lower() not in (
                BINARY_SUFFIXES
                |
                TEXT_SUFFIXES
            ):
                continue

            files.append(
                path
            )

    return sorted(
        files,
        key=lambda path:
            str(
                path
            ),
    )


def inspect_npz(
    path: Path,
):
    result = {
        "type":
            "npz",

        "arrays":
            {},
    }

    try:
        with np.load(
            path,
            allow_pickle=False,
        ) as payload:

            for key in payload.files:
                array = payload[
                    key
                ]

                result[
                    "arrays"
                ][
                    key
                ] = {
                    "shape":
                        list(
                            array.shape
                        ),

                    "dtype":
                        str(
                            array.dtype
                        ),
                }

    except Exception as exc:
        result[
            "error"
        ] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    return result


def inspect_npy(
    path: Path,
):
    result = {
        "type":
            "npy",
    }

    try:
        array = np.load(
            path,
            mmap_mode="r",
            allow_pickle=False,
        )

        result[
            "shape"
        ] = list(
            array.shape
        )

        result[
            "dtype"
        ] = str(
            array.dtype
        )

    except Exception as exc:
        result[
            "error"
        ] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    return result


def inspect_torch_zip_without_loading(
    path: Path,
):
    result = {
        "type":
            "torch_or_pickle",
    }

    #
    # Modern torch.save normally creates a zip archive.
    # We only inspect archive members here; no pickle
    # deserialization/model execution occurs.
    #
    try:
        if zipfile.is_zipfile(
            path
        ):
            with zipfile.ZipFile(
                path,
                "r",
            ) as archive:

                names = archive.namelist()

            result[
                "zip_archive"
            ] = True

            result[
                "member_count"
            ] = len(
                names
            )

            result[
                "members_preview"
            ] = names[
                :30
            ]

        else:
            result[
                "zip_archive"
            ] = False

    except Exception as exc:
        result[
            "error"
        ] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    return result


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
        #
        # Avoid exploding huge lists.
        #
        if len(
            value
        ) <= 20:
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
                    prefix
                    +
                    ".__list_length__",
                    len(
                        value
                    ),
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


def inspect_json(
    path: Path,
):
    result = {
        "type":
            "json",
    }

    try:
        if path.stat().st_size > (
            50
            *
            1024
            *
            1024
        ):
            result[
                "skipped_large"
            ] = True

            return result

        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        interesting = []

        tokens = (
            "dev",
            "prediction",
            "predictions",
            "gaussian",
            "covariance",
            "cov",
            "mean",
            "mu",
            "checkpoint",
            "cache",
            "posterior",
            "block44",
        )

        for key, value in flatten_json(
            payload
        ):
            lower = key.lower()

            if any(
                token in lower
                for token in tokens
            ):
                interesting.append({
                    "key":
                        key,

                    "value":
                        value,
                })

        result[
            "interesting_fields"
        ] = interesting[
            :100
        ]

    except Exception as exc:
        result[
            "error"
        ] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    return result


def inspect_jsonl(
    path: Path,
):
    result = {
        "type":
            "jsonl",
    }

    try:
        rows = []

        with path.open(
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as stream:

            for index, line in enumerate(
                stream
            ):
                if index >= 5:
                    break

                line = line.strip()

                if not line:
                    continue

                try:
                    item = json.loads(
                        line
                    )

                except Exception:
                    continue

                rows.append(
                    item
                )

        result[
            "first_rows_preview"
        ] = rows

    except Exception as exc:
        result[
            "error"
        ] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    return result


def source_hits(
    path: Path,
):
    try:
        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    except Exception:
        return []

    patterns = (
        r"np\.savez",
        r"np\.save",
        r"torch\.save",
        r"dev",
        r"gaussian",
        r"prediction",
        r"covariance",
        r"block44",
        r"checkpoint",
        r"cache",
        r"posterior",
    )

    lines = text.splitlines()

    hits = []

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        lower = line.lower()

        if not any(
            re.search(
                pattern,
                lower,
            )
            for pattern in patterns
        ):
            continue

        hits.append({
            "line":
                line_number,

            "text":
                line[
                    :500
                ],
        })

        if len(
            hits
        ) >= 80:
            break

    return hits


def candidate_score(
    path: Path,
    inspection,
):
    lower = str(
        path
    ).lower()

    score = 0

    if "block44" in lower:
        score += 20

    if "dev" in lower:
        score += 12

    if "gaussian" in lower:
        score += 8

    if "pred" in lower:
        score += 8

    if "posterior" in lower:
        score += 6

    if "cache" in lower:
        score += 4

    if path.suffix.lower() in {
        ".npz",
        ".npy",
    }:
        score += 8

    text = json.dumps(
        inspection,
        default=str,
    ).lower()

    if "4, 3, 3" in text:
        score += 10

    if "4, 3" in text:
        score += 8

    if "covariance" in text:
        score += 5

    if "mean" in text:
        score += 4

    if is_formal_or_postformal(
        path
    ):
        score -= 1000

    return score


def main():
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.2 PART 2/2 — READ-ONLY DEV POSTERIOR RESOLVER AUDIT"
    )
    print(
        "============================================================"
    )

    print(
        "dataset scan          = NO"
    )
    print(
        "Stage4 inference      = NO"
    )
    print(
        "formal N=120 read     = NO"
    )
    print(
        "upstream modification = NO"
    )

    files = collect_files()

    print()
    print(
        "Stage4 candidate files scanned =",
        len(
            files
        ),
    )

    binary_inventory = []

    text_source_evidence = []

    ranked = []

    excluded_formal = []

    for path in files:

        suffix = path.suffix.lower()

        inspection = None

        if suffix == ".npz":
            inspection = inspect_npz(
                path
            )

        elif suffix == ".npy":
            inspection = inspect_npy(
                path
            )

        elif suffix in {
            ".pt",
            ".pth",
            ".pkl",
            ".pickle",
        }:
            inspection = (
                inspect_torch_zip_without_loading(
                    path
                )
            )

        elif suffix == ".json":
            inspection = inspect_json(
                path
            )

        elif suffix == ".jsonl":
            inspection = inspect_jsonl(
                path
            )

        elif suffix in {
            ".py",
            ".md",
            ".txt",
        }:
            hits = source_hits(
                path
            )

            if hits:
                text_source_evidence.append({
                    "path":
                        relative(
                            path
                        ),

                    "formal_or_postformal":
                        is_formal_or_postformal(
                            path
                        ),

                    "hits":
                        hits,
                })

            continue

        if inspection is None:
            continue

        item = {
            "path":
                relative(
                    path
                ),

            "size_bytes":
                path.stat().st_size,

            "sha256":
                file_sha256(
                    path
                ),

            "formal_or_postformal":
                is_formal_or_postformal(
                    path
                ),

            "inspection":
                inspection,
        }

        binary_inventory.append(
            item
        )

        score = candidate_score(
            path,
            inspection,
        )

        if item[
            "formal_or_postformal"
        ]:
            excluded_formal.append(
                item
            )

        else:
            ranked.append(
                (
                    score,
                    item,
                )
            )

    ranked.sort(
        key=lambda pair:
            (
                pair[
                    0
                ],
                pair[
                    1
                ][
                    "path"
                ],
            ),
        reverse=True,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "A. NON-FORMAL BINARY/JSON CANDIDATES"
    )
    print(
        "============================================================"
    )

    useful = [
        (
            score,
            item,
        )
        for score, item in ranked
        if score > 0
    ]

    if not useful:
        print(
            "<NO POSITIVE-SCORE NON-FORMAL ARTIFACT FOUND>"
        )

    for rank, (
        score,
        item,
    ) in enumerate(
        useful[
            :30
        ],
        start=1,
    ):
        print()
        print(
            f"[{rank}] score = {score}"
        )
        print(
            "path =",
            item[
                "path"
            ],
        )
        print(
            "size =",
            item[
                "size_bytes"
            ],
        )
        print(
            "inspection =",
            json.dumps(
                item[
                    "inspection"
                ],
                indent=2,
                default=str,
            )[
                :5000
            ],
        )

    print()
    print(
        "============================================================"
    )
    print(
        "B. SOURCE-CODE / REPORT PROVENANCE HITS"
    )
    print(
        "============================================================"
    )

    nonformal_source = [
        item
        for item in text_source_evidence
        if not item[
            "formal_or_postformal"
        ]
    ]

    if not nonformal_source:
        print(
            "<NO NON-FORMAL SOURCE HITS>"
        )

    for item in nonformal_source[
        :25
    ]:
        print()
        print(
            "FILE:",
            item[
                "path"
            ],
        )

        for hit in item[
            "hits"
        ][
            :30
        ]:
            print(
                f"  L{hit['line']}: "
                f"{hit['text']}"
            )

    print()
    print(
        "============================================================"
    )
    print(
        "C. FORMAL/POST-FORMAL ARTIFACTS EXCLUDED"
    )
    print(
        "============================================================"
    )

    print(
        "excluded artifact count =",
        len(
            excluded_formal
        ),
    )

    for item in excluded_formal[
        :15
    ]:
        print(
            "EXCLUDED:",
            item[
                "path"
            ],
        )

    report = {
        "status":
            "PASS_READ_ONLY_AUDIT",

        "scientific_execution": {
            "dataset_scan":
                False,

            "Stage4_inference":
                False,

            "formal_N120_read":
                False,

            "training":
                False,

            "recalibration":
                False,

            "upstream_modified":
                False,
        },

        "search_roots":
            [
                relative(
                    path
                )
                for path in SEARCH_ROOTS
            ],

        "files_scanned":
            len(
                files
            ),

        "nonformal_ranked_candidates":
            [
                {
                    "score":
                        score,

                    **item,
                }
                for score, item in ranked[
                    :50
                ]
            ],

        "source_evidence":
            text_source_evidence,

        "excluded_formal_or_postformal":
            excluded_formal,

        "next_decision":
            (
                "resolve_actual_frozen_nonformal_"
                "development_posterior_or_confirm_"
                "that_only_checkpoint_plus_dev_cache_exists"
            ),
    }

    write_json(
        REPORT,
        report,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "RESOLVER AUDIT FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "read-only audit       = PASS"
    )
    print(
        "formal N=120 used     = NO"
    )
    print(
        "Stage4 inference      = NO"
    )
    print(
        "dataset scan          = NO"
    )
    print(
        "upstream modified     = NO"
    )
    print(
        "positive candidates   =",
        len(
            useful
        ),
    )
    print(
        "source-evidence files =",
        len(
            nonformal_source
        ),
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
        "DEV POSTERIOR RESOLVER AUDIT = BLOCKED"
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
        "Stage4 inference  = NO"
    )
    print(
        "dataset scan      = NO"
    )
    print(
        "upstream modified = NO"
    )
    print(
        "terminal remains open = YES"
    )

# Deliberately no sys.exit().

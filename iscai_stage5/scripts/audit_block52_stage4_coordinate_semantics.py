from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
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

NORMALIZATION = (
    STAGE4
    / "artifacts/block43/"
      "fit_normalization.json"
)

DEV_CACHE = (
    STAGE4
    / "artifacts/block43/"
      "development_cache.npz"
)

BLOCK44_REPORT = (
    STAGE4
    / "reports/"
      "block44_gaussian_gru.json"
)

REPORT = (
    STAGE5
    / "artifacts/block52/"
      "stage4_prediction_coordinate_semantics.json"
)


SEARCH_ROOTS = (
    STAGE4
    / "src",

    STAGE4
    / "scripts",

    STAGE4
    / "configs",

    STAGE4
    / "reports",
)


TOKENS = (
    "development_cache.npz",
    "fit_cache.npz",
    "fit_normalization",
    "label_mean",
    "label_std",
    "future",
    "future_mask",
    "displacement",
    "relative",
    "anchor",
    "current_position",
    "current position",
    "target_position",
    "target position",
    "h0",
    "denormalize_gaussian",
)


STRONG_TOKENS = (
    "development_cache.npz",
    "label_mean",
    "label_std",
    "displacement",
    "future_mask",
    "denormalize_gaussian",
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


def source_score(
    path: Path,
    text: str,
):
    lower_path = str(
        path
    ).lower()

    lower = text.lower()

    score = 0

    if "block43" in lower_path:
        score += 100

    if "block42" in lower_path:
        score += 70

    if "dataset" in lower_path:
        score += 40

    if "cache" in lower_path:
        score += 30

    if "training" in lower_path:
        score += 20

    if "normal" in lower_path:
        score += 20

    for token in TOKENS:

        if token.lower() in lower:
            score += 2

    for token in STRONG_TOKENS:

        if token.lower() in lower:
            score += 10

    return score


def merge_line_numbers(
    numbers,
    *,
    radius=8,
):
    if not numbers:
        return []

    numbers = sorted(
        set(
            numbers
        )
    )

    groups = []

    current = [
        numbers[
            0
        ]
    ]

    for value in numbers[
        1:
    ]:

        if (
            value
            -
            current[
                -1
            ]
        ) <= (
            2
            *
            radius
            +
            1
        ):
            current.append(
                value
            )

        else:
            groups.append(
                current
            )

            current = [
                value
            ]

    groups.append(
        current
    )

    return groups


def source_windows(
    path: Path,
    *,
    radius=8,
    max_windows=8,
):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    lines = text.splitlines()

    matched = []

    for index, line in enumerate(
        lines
    ):
        lower = line.lower()

        if any(
            token.lower()
            in lower
            for token in TOKENS
        ):
            matched.append(
                index
            )

    groups = merge_line_numbers(
        matched,
        radius=radius,
    )

    windows = []

    for group in groups[
        :max_windows
    ]:

        start = max(
            0,
            min(
                group
            )
            -
            radius,
        )

        end = min(
            len(
                lines
            ),
            max(
                group
            )
            +
            radius
            +
            1,
        )

        windows.append({
            "start_line":
                start
                +
                1,

            "end_line":
                end,

            "lines":
                [
                    {
                        "line":
                            line_index
                            +
                            1,

                        "text":
                            lines[
                                line_index
                            ],
                    }
                    for line_index in range(
                        start,
                        end,
                    )
                ],
        })

    return windows


def collect_sources():
    items = []

    suffixes = {
        ".py",
        ".json",
        ".md",
        ".txt",
    }

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

            if path.suffix.lower() not in suffixes:
                continue

            try:
                if path.stat().st_size > (
                    20
                    *
                    1024
                    *
                    1024
                ):
                    continue

                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )

            except Exception:
                continue

            lower = text.lower()

            hit_tokens = [
                token
                for token in TOKENS
                if token.lower()
                in lower
            ]

            if not hit_tokens:
                continue

            items.append({
                "path":
                    path,

                "score":
                    source_score(
                        path,
                        text,
                    ),

                "hit_tokens":
                    hit_tokens,

                "windows":
                    source_windows(
                        path
                    ),
            })

    items.sort(
        key=lambda item:
            (
                item[
                    "score"
                ],
                str(
                    item[
                        "path"
                    ]
                ),
            ),
        reverse=True,
    )

    return items


def load_normalization():
    payload = json.loads(
        NORMALIZATION.read_text(
            encoding="utf-8"
        )
    )

    return payload


def print_windows(
    item,
):
    print()
    print(
        "FILE =",
        relative(
            item[
                "path"
            ]
        ),
    )

    print(
        "score =",
        item[
            "score"
        ],
    )

    print(
        "tokens =",
        item[
            "hit_tokens"
        ],
    )

    for index, window in enumerate(
        item[
            "windows"
        ],
        start=1,
    ):
        print()
        print(
            f"--- window {index}: "
            f"L{window['start_line']}-"
            f"L{window['end_line']} ---"
        )

        for line in window[
            "lines"
        ]:
            print(
                f"{line['line']:5d}: "
                f"{line['text']}"
            )


def semantic_signal(
    sources,
):
    """
    This deliberately does NOT infer semantics from vague
    words alone.

    It only records explicit source-level signals for the
    next scientific decision.
    """

    absolute_signals = []

    displacement_signals = []

    anchor_signals = []

    for item in sources:

        for window in item[
            "windows"
        ]:

            for line in window[
                "lines"
            ]:

                text = str(
                    line[
                        "text"
                    ]
                )

                lower = text.lower()

                evidence = {
                    "path":
                        relative(
                            item[
                                "path"
                            ]
                        ),

                    "line":
                        line[
                            "line"
                        ],

                    "text":
                        text[
                            :500
                        ],
                }

                if any(
                    token in lower
                    for token in (
                        "future -",
                        "future=",
                        "displacement",
                        "relative to",
                        "relative_to",
                        "delta_position",
                        "delta position",
                    )
                ):
                    displacement_signals.append(
                        evidence
                    )

                if any(
                    token in lower
                    for token in (
                        "absolute",
                        "world position",
                        "h0 position",
                        "position_h0",
                    )
                ):
                    absolute_signals.append(
                        evidence
                    )

                if any(
                    token in lower
                    for token in (
                        "anchor",
                        "current_position",
                        "current position",
                        "last_observed",
                        "last observed",
                    )
                ):
                    anchor_signals.append(
                        evidence
                    )

    return {
        "displacement_signals":
            displacement_signals[
                :100
            ],

        "absolute_signals":
            absolute_signals[
                :100
            ],

        "anchor_signals":
            anchor_signals[
                :100
            ],
    }


def main():
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.2 — STAGE4 PREDICTION-COORDINATE SEMANTIC AUDIT"
    )
    print(
        "============================================================"
    )

    required = (
        NORMALIZATION,
        DEV_CACHE,
        BLOCK44_REPORT,
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
            "Missing frozen prerequisite(s): "
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
    # A. Frozen normalization
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "A. FROZEN BLOCK4.3 NORMALIZATION"
    )
    print(
        "============================================================"
    )

    normalization = (
        load_normalization()
    )

    normalization_sha = (
        file_sha256(
            NORMALIZATION
        )
    )

    print(
        "path =",
        NORMALIZATION,
    )

    print(
        "SHA256 =",
        normalization_sha,
    )

    require(
        normalization_sha
        ==
        (
            "3d7fc0a66d4a4f566f6569befa9c3766"
            "ecae21830bb4e46256df2f326a82a5f6"
        ),
        (
            "Frozen Stage4 normalization "
            "SHA changed."
        ),
    )

    print(
        json.dumps(
            normalization,
            indent=2,
            sort_keys=True,
            default=str,
        )[
            :16000
        ]
    )

    # ========================================================
    # B. Exact source search
    # ========================================================

    sources = collect_sources()

    print()
    print(
        "============================================================"
    )
    print(
        "B. TOP STAGE4 FUTURE/CACHE/NORMALIZATION SOURCE ROUTES"
    )
    print(
        "============================================================"
    )

    print(
        "relevant source files =",
        len(
            sources
        ),
    )

    for item in sources[
        :12
    ]:
        print_windows(
            item
        )

    # ========================================================
    # C. Explicit semantic evidence only
    # ========================================================

    signals = semantic_signal(
        sources
    )

    print()
    print(
        "============================================================"
    )
    print(
        "C. EXPLICIT COORDINATE-SEMANTIC SIGNALS"
    )
    print(
        "============================================================"
    )

    print()
    print(
        "DISPLACEMENT / RELATIVE SIGNALS"
    )

    for item in signals[
        "displacement_signals"
    ][
        :30
    ]:
        print(
            f"{item['path']}:"
            f"L{item['line']}: "
            f"{item['text']}"
        )

    print()
    print(
        "ABSOLUTE / H0-POSITION SIGNALS"
    )

    for item in signals[
        "absolute_signals"
    ][
        :30
    ]:
        print(
            f"{item['path']}:"
            f"L{item['line']}: "
            f"{item['text']}"
        )

    print()
    print(
        "ANCHOR / CURRENT-POSITION SIGNALS"
    )

    for item in signals[
        "anchor_signals"
    ][
        :30
    ]:
        print(
            f"{item['path']}:"
            f"L{item['line']}: "
            f"{item['text']}"
        )

    # ========================================================
    # D. Freeze audit evidence
    # ========================================================

    report_payload = {
        "status":
            "PASS_READ_ONLY_SEMANTIC_AUDIT",

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

        "normalization": {
            "path":
                str(
                    NORMALIZATION
                ),

            "sha256":
                normalization_sha,

            "payload":
                normalization,
        },

        "development_cache": {
            "path":
                str(
                    DEV_CACHE
                ),

            "sha256":
                file_sha256(
                    DEV_CACHE
                ),
        },

        "source_routes":
            [
                {
                    "path":
                        relative(
                            item[
                                "path"
                            ]
                        ),

                    "score":
                        item[
                            "score"
                        ],

                    "hit_tokens":
                        item[
                            "hit_tokens"
                        ],

                    "windows":
                        item[
                            "windows"
                        ],
                }
                for item in sources[
                    :30
                ]
            ],

        "semantic_signals":
            signals,

        "scientific_decision":
            (
                "DO_NOT_RUN_DEV_INFERENCE_UNTIL_"
                "FUTURE_COORDINATE_SEMANTICS_ARE_RESOLVED"
            ),
    }

    write_json(
        REPORT,
        report_payload,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "SEMANTIC AUDIT FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "normalization SHA      = PASS"
    )

    print(
        "source evidence        = COLLECTED"
    )

    print(
        "model forward executed = NO"
    )

    print(
        "formal N=120 used      = NO"
    )

    print(
        "upstream modified      = NO"
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
        "STAGE4 COORDINATE SEMANTIC AUDIT = BLOCKED"
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

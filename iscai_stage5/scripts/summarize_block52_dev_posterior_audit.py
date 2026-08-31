from __future__ import annotations

import json
from pathlib import Path
import traceback


ROOT = Path(
    "/home/agni/waymo"
)

STAGE5 = (
    ROOT
    / "iscai_stage5"
)

AUDIT = (
    STAGE5
    / "artifacts/block52/"
      "dev_posterior_resolver_audit.json"
)

SUMMARY = (
    STAGE5
    / "artifacts/block52/"
      "dev_posterior_resolver_summary.json"
)


IMPORTANT_PATH_TOKENS = (
    "block44",
    "block43",
    "block45",
    "gaussian",
    "dev",
    "prediction",
    "posterior",
    "cache",
)

STRONG_PATH_TOKENS = (
    "block44",
    "gaussian",
    "dev",
    "prediction",
    "posterior",
)

IMPORTANT_HIT_TOKENS = (
    "np.save",
    "np.savez",
    "torch.save",
    "gaussian",
    "dev",
    "prediction",
    "posterior",
    "covariance",
    "checkpoint",
    "cache",
    "block44",
)


def load_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def write_json(
    path: Path,
    payload,
):
    path.write_text(
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


def relevant_candidate(
    item,
):
    path = str(
        item.get(
            "path",
            ""
        )
    ).lower()

    inspection = json.dumps(
        item.get(
            "inspection",
            {}
        ),
        default=str,
    ).lower()

    path_hits = sum(
        token in path
        for token in IMPORTANT_PATH_TOKENS
    )

    strong_hits = sum(
        token in path
        for token in STRONG_PATH_TOKENS
    )

    shape_signal = (
        "4, 3, 3"
        in inspection
        or
        "[4, 3, 3]"
        in inspection
        or
        "4, 3"
        in inspection
        or
        "[4, 3]"
        in inspection
    )

    covariance_signal = (
        "cov"
        in inspection
        or
        "sigma"
        in inspection
    )

    mean_signal = (
        "mean"
        in inspection
        or
        '"mu"'
        in inspection
        or
        "pred"
        in inspection
    )

    return (
        strong_hits >= 1
        or
        path_hits >= 2
        or
        (
            shape_signal
            and
            covariance_signal
            and
            mean_signal
        )
    )


def candidate_priority(
    item,
):
    path = str(
        item.get(
            "path",
            ""
        )
    ).lower()

    inspection = json.dumps(
        item.get(
            "inspection",
            {}
        ),
        default=str,
    ).lower()

    score = int(
        item.get(
            "score",
            0
        )
    )

    if "block44" in path:
        score += 100

    if "dev" in path:
        score += 60

    if "gaussian" in path:
        score += 40

    if "prediction" in path or "pred" in path:
        score += 30

    if "posterior" in path:
        score += 25

    if path.endswith(
        ".npz"
    ):
        score += 20

    if path.endswith(
        ".npy"
    ):
        score += 15

    if "4, 3, 3" in inspection:
        score += 30

    if "[4, 3, 3]" in inspection:
        score += 30

    if "4, 3" in inspection:
        score += 15

    if "[4, 3]" in inspection:
        score += 15

    return score


def relevant_source(
    item,
):
    path = str(
        item.get(
            "path",
            ""
        )
    ).lower()

    if item.get(
        "formal_or_postformal"
    ):
        return False

    if any(
        token in path
        for token in (
            "block44",
            "gaussian",
            "train",
            "evaluate",
            "eval",
            "predict",
            "inference",
            "model",
            "dataset",
            "cache",
        )
    ):
        return True

    hits = item.get(
        "hits",
        []
    )

    for hit in hits:
        text = str(
            hit.get(
                "text",
                ""
            )
        ).lower()

        if any(
            token in text
            for token in IMPORTANT_HIT_TOKENS
        ):
            return True

    return False


def source_priority(
    item,
):
    path = str(
        item.get(
            "path",
            ""
        )
    ).lower()

    score = 0

    if "block44" in path:
        score += 100

    if "gaussian" in path:
        score += 50

    if "train" in path:
        score += 30

    if "predict" in path:
        score += 30

    if "eval" in path:
        score += 20

    if "cache" in path:
        score += 20

    for hit in item.get(
        "hits",
        []
    ):
        text = str(
            hit.get(
                "text",
                ""
            )
        ).lower()

        if "np.savez" in text:
            score += 40

        if "np.save" in text:
            score += 25

        if "torch.save" in text:
            score += 20

        if "dev" in text:
            score += 10

        if "prediction" in text:
            score += 10

        if "covariance" in text:
            score += 10

    return score


def print_candidate(
    rank,
    item,
):
    print()
    print(
        f"[CANDIDATE {rank}]"
    )

    print(
        "priority =",
        candidate_priority(
            item
        ),
    )

    print(
        "original score =",
        item.get(
            "score"
        ),
    )

    print(
        "path =",
        item.get(
            "path"
        ),
    )

    print(
        "size =",
        item.get(
            "size_bytes"
        ),
    )

    print(
        "sha256 =",
        item.get(
            "sha256"
        ),
    )

    inspection = item.get(
        "inspection",
        {}
    )

    print(
        "inspection ="
    )

    print(
        json.dumps(
            inspection,
            indent=2,
            default=str,
        )[
            :6000
        ]
    )


def print_source(
    rank,
    item,
):
    print()
    print(
        f"[SOURCE {rank}]"
    )

    print(
        "priority =",
        source_priority(
            item
        ),
    )

    print(
        "path =",
        item.get(
            "path"
        ),
    )

    hits = item.get(
        "hits",
        []
    )

    for hit in hits[
        :25
    ]:
        text = str(
            hit.get(
                "text",
                ""
            )
        )

        lower = text.lower()

        if not any(
            token in lower
            for token in IMPORTANT_HIT_TOKENS
        ):
            continue

        print(
            f"L{hit.get('line')}: "
            f"{text}"
        )


def main():
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.2 — TARGETED DEV POSTERIOR AUDIT SUMMARY"
    )
    print(
        "============================================================"
    )

    if not AUDIT.is_file():
        raise RuntimeError(
            f"Audit JSON missing: {AUDIT}"
        )

    payload = load_json(
        AUDIT
    )

    print(
        "original audit status =",
        payload.get(
            "status"
        ),
    )

    execution = payload.get(
        "scientific_execution",
        {}
    )

    print(
        "formal N=120 read      =",
        execution.get(
            "formal_N120_read"
        ),
    )

    print(
        "Stage4 inference       =",
        execution.get(
            "Stage4_inference"
        ),
    )

    print(
        "dataset scan           =",
        execution.get(
            "dataset_scan"
        ),
    )

    print(
        "upstream modified      =",
        execution.get(
            "upstream_modified"
        ),
    )

    candidates = [
        item
        for item in payload.get(
            "nonformal_ranked_candidates",
            []
        )
        if relevant_candidate(
            item
        )
    ]

    candidates.sort(
        key=lambda item:
            (
                candidate_priority(
                    item
                ),
                str(
                    item.get(
                        "path",
                        ""
                    )
                ),
            ),
        reverse=True,
    )

    sources = [
        item
        for item in payload.get(
            "source_evidence",
            []
        )
        if relevant_source(
            item
        )
    ]

    sources.sort(
        key=lambda item:
            (
                source_priority(
                    item
                ),
                str(
                    item.get(
                        "path",
                        ""
                    )
                ),
            ),
        reverse=True,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "A. TOP RELEVANT NON-FORMAL ARTIFACT CANDIDATES"
    )
    print(
        "============================================================"
    )

    print(
        "relevant candidates =",
        len(
            candidates
        ),
    )

    for rank, item in enumerate(
        candidates[
            :15
        ],
        start=1,
    ):
        print_candidate(
            rank,
            item,
        )

    print()
    print(
        "============================================================"
    )
    print(
        "B. TOP RELEVANT SOURCE / REPORT PROVENANCE"
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

    for rank, item in enumerate(
        sources[
            :15
        ],
        start=1,
    ):
        print_source(
            rank,
            item,
        )

    #
    # Extra compact Block4.4-focused inventory.
    #
    block44_candidates = [
        item
        for item in payload.get(
            "nonformal_ranked_candidates",
            []
        )
        if "block44"
        in
        str(
            item.get(
                "path",
                ""
            )
        ).lower()
    ]

    block44_sources = [
        item
        for item in payload.get(
            "source_evidence",
            []
        )
        if (
            not item.get(
                "formal_or_postformal"
            )
            and
            "block44"
            in
            str(
                item.get(
                    "path",
                    ""
                )
            ).lower()
        )
    ]

    print()
    print(
        "============================================================"
    )
    print(
        "C. EXACT BLOCK4.4-FOCUSED SUMMARY"
    )
    print(
        "============================================================"
    )

    print(
        "Block4.4 non-formal artifacts =",
        len(
            block44_candidates
        ),
    )

    for item in block44_candidates[
        :20
    ]:
        print(
            "ARTIFACT:",
            item.get(
                "path"
            ),
        )

        inspection = item.get(
            "inspection",
            {}
        )

        print(
            "  inspection =",
            json.dumps(
                inspection,
                default=str,
            )[
                :2000
            ],
        )

    print(
        "Block4.4 source files =",
        len(
            block44_sources
        ),
    )

    for item in block44_sources[
        :20
    ]:
        print(
            "SOURCE:",
            item.get(
                "path"
            ),
        )

        for hit in item.get(
            "hits",
            []
        )[
            :20
        ]:
            print(
                f"  L{hit.get('line')}: "
                f"{hit.get('text')}"
            )

    summary = {
        "status":
            "PASS_TARGETED_EXTRACTION",

        "source_audit":
            str(
                AUDIT
            ),

        "scientific_execution":
            execution,

        "top_candidates":
            candidates[
                :30
            ],

        "top_sources":
            sources[
                :30
            ],

        "block44_candidates":
            block44_candidates,

        "block44_sources":
            block44_sources,
    }

    write_json(
        SUMMARY,
        summary,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "TARGETED SUMMARY FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "status               = PASS"
    )

    print(
        "formal N=120 used     = NO"
    )

    print(
        "Stage4 inference      = NO"
    )

    print(
        "filesystem rescan     = NO"
    )

    print(
        "top candidates shown  =",
        min(
            15,
            len(
                candidates
            ),
        ),
    )

    print(
        "top sources shown     =",
        min(
            15,
            len(
                sources
            ),
        ),
    )

    print(
        "summary =",
        SUMMARY,
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
        "TARGETED SUMMARY = BLOCKED"
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

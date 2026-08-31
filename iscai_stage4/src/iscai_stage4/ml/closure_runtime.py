from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re


HEX64 = re.compile(
    r"^[0-9a-f]{64}$"
)


def file_sha256(
    path,
):
    path = Path(
        path
    )

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


def recursive_values(
    value,
):
    if isinstance(
        value,
        dict,
    ):
        for item in value.values():
            yield from recursive_values(
                item
            )

    elif isinstance(
        value,
        list,
    ):
        for item in value:
            yield from recursive_values(
                item
            )

    else:
        yield value


def recursive_hex64_values(
    value,
):
    result = []

    for item in recursive_values(
        value
    ):
        if (
            isinstance(
                item,
                str,
            )
            and
            HEX64.fullmatch(
                item
            )
        ):
            result.append(
                item
            )

    return tuple(
        result
    )


def report_contains_sha(
    payload,
    expected_sha,
):
    return (
        str(
            expected_sha
        )
        in
        recursive_hex64_values(
            payload
        )
    )


def discover_frozen_block_report(
    reports_dir,
    *,
    block,
    expected_implementation_sha,
):
    """
    Locate the final frozen PASS report without
    depending on historical filename conventions.

    Strong requirements:
      - JSON parses
      - top-level status == PASS
      - top-level block identifies requested block
      - frozen implementation SHA occurs in payload
    """

    reports_dir = Path(
        reports_dir
    )

    candidates = []

    for path in sorted(
        reports_dir.glob(
            "*.json"
        )
    ):
        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

        except Exception:
            continue

        if (
            payload.get(
                "status"
            )
            !=
            "PASS"
        ):
            continue

        declared_block = str(
            payload.get(
                "block",
                "",
            )
        )

        if not (
            declared_block
            ==
            block
            or
            declared_block.startswith(
                block
                +
                "_"
            )
        ):
            continue

        if not report_contains_sha(
            payload,
            expected_implementation_sha,
        ):
            continue

        score = (
            2
            if declared_block
            ==
            block
            else
            1
        )

        candidates.append(
            (
                score,
                path,
                payload,
            )
        )

    if not candidates:
        raise RuntimeError(
            "Could not discover frozen PASS "
            f"report for Block {block} with "
            f"implementation SHA "
            f"{expected_implementation_sha}."
        )

    best_score = max(
        item[
            0
        ]
        for item in candidates
    )

    best = [
        item
        for item in candidates
        if item[
            0
        ]
        ==
        best_score
    ]

    if len(
        best
    ) != 1:
        raise RuntimeError(
            "Ambiguous frozen report discovery "
            f"for Block {block}: "
            +
            repr(
                [
                    str(
                        item[
                            1
                        ]
                    )
                    for item in best
                ]
            )
        )

    (
        _,
        path,
        payload,
    ) = best[
        0
    ]

    return {
        "path":
            path,

        "payload":
            payload,

        "file_sha256":
            file_sha256(
                path
            ),
    }


def require_true(
    value,
    *,
    name,
):
    if not bool(
        value
    ):
        raise RuntimeError(
            f"Mandatory closure criterion "
            f"is not PASS: {name}"
        )

    return True


def require_false(
    value,
    *,
    name,
):
    if bool(
        value
    ):
        raise RuntimeError(
            f"Forbidden closure condition "
            f"is true: {name}"
        )

    return True


def acceptance_matrix(
    criteria,
):
    """
    `criteria`:
       iterable of dicts with:
         id, mandatory, pass, evidence

    Mandatory failure blocks closure.
    Non-mandatory diagnostics remain visible
    but cannot silently become requirements.
    """

    rows = []

    mandatory_failures = []

    for criterion in criteria:
        row = {
            "id":
                str(
                    criterion[
                        "id"
                    ]
                ),

            "mandatory":
                bool(
                    criterion[
                        "mandatory"
                    ]
                ),

            "pass":
                bool(
                    criterion[
                        "pass"
                    ]
                ),

            "evidence":
                criterion.get(
                    "evidence"
                ),
        }

        rows.append(
            row
        )

        if (
            row[
                "mandatory"
            ]
            and
            not
            row[
                "pass"
            ]
        ):
            mandatory_failures.append(
                row[
                    "id"
                ]
            )

    return {
        "rows":
            rows,

        "mandatory_count":
            sum(
                1
                for row in rows
                if row[
                    "mandatory"
                ]
            ),

        "mandatory_pass_count":
            sum(
                1
                for row in rows
                if (
                    row[
                        "mandatory"
                    ]
                    and
                    row[
                        "pass"
                    ]
                )
            ),

        "mandatory_failures":
            mandatory_failures,

        "closure_ready":
            (
                len(
                    mandatory_failures
                )
                ==
                0
            ),
    }

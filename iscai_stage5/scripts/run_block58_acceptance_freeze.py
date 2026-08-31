from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(
    "/home/agni/waymo"
)

STAGE5 = (
    ROOT
    /
    "iscai_stage5"
)

sys.path.insert(
    0,
    str(
        STAGE5
        /
        "src"
    ),
)

from iscai_stage5.formal_acceptance import (
    FORMAL_EMPIRICAL_ALPHA,
    FORMAL_COVERAGE_METHOD,
    NOMINAL_COVERAGE_Q,
    REQUESTED_COVERAGE_LEVELS,
    formal_acceptance_policy_dict,
)


ROUTE_AUDIT = (
    STAGE5
    /
    "artifacts/block58/"
    "formal_input_route_audit.json"
)

FORMAL_MANIFEST = (
    ROOT
    /
    "iscai_stage3/artifacts/block38e/"
    "formal_validation_120.jsonl"
)

EXPECTED_FORMAL_MANIFEST_SHA = (
    "2208e7287ddf6439fda4597c435a9cba"
    "1d1b9d0e4c4547bc5dd92e56e8124e46"
)

POLICY = (
    STAGE5
    /
    "configs/"
    "formal_stage5_acceptance_policy.json"
)

REPORT = (
    STAGE5
    /
    "reports/"
    "block58_prefreeze_acceptance.json"
)


def file_sha256(
    path: Path,
):
    digest = sha256()

    with path.open(
        "rb"
    ) as stream:
        while True:
            block = stream.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def canonical_bytes(
    payload,
):
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        +
        "\n"
    ).encode(
        "utf-8"
    )


def scalar_values(
    value,
):
    result = []

    if isinstance(
        value,
        dict,
    ):
        for item in value.values():
            result.extend(
                scalar_values(
                    item
                )
            )

    elif isinstance(
        value,
        list,
    ):
        for item in value:
            result.extend(
                scalar_values(
                    item
                )
            )

    else:
        result.append(
            value
        )

    return result


def atomic_write(
    path: Path,
    data: bytes,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = (
        path
        .with_suffix(
            path.suffix
            +
            ".tmp"
        )
    )

    temporary.write_bytes(
        data
    )

    temporary.replace(
        path
    )


def main():
    if not ROUTE_AUDIT.is_file():
        raise RuntimeError(
            "Block5.8 Part1 route audit is missing."
        )

    route = json.loads(
        ROUTE_AUDIT.read_text(
            encoding="utf-8"
        )
    )

    values = {
        str(value)
        for value in scalar_values(
            route
        )
    }

    if (
        "PASS_READ_ONLY_AUDIT"
        not in values
    ):
        raise RuntimeError(
            "Block5.8 Part1 PASS evidence missing."
        )

    if (
        "READY_FOR_EXACT_ROUTE_SELECTION"
        not in values
    ):
        raise RuntimeError(
            "Block5.8 route-selection readiness "
            "evidence missing."
        )

    if not FORMAL_MANIFEST.is_file():
        raise RuntimeError(
            "Frozen formal manifest is missing."
        )

    manifest_sha = (
        file_sha256(
            FORMAL_MANIFEST
        )
    )

    if (
        manifest_sha
        !=
        EXPECTED_FORMAL_MANIFEST_SHA
    ):
        raise RuntimeError(
            "Frozen formal manifest SHA changed."
        )

    policy = (
        formal_acceptance_policy_dict()
    )

    encoded = (
        canonical_bytes(
            policy
        )
    )

    if POLICY.is_file():
        existing = (
            POLICY.read_bytes()
        )

        if (
            existing
            !=
            encoded
        ):
            raise RuntimeError(
                "Existing formal acceptance policy "
                "differs from the pre-formal freeze. "
                "Refusing post-hoc overwrite."
            )
    else:
        atomic_write(
            POLICY,
            encoded,
        )

    policy_sha = (
        file_sha256(
            POLICY
        )
    )

    report = {
        "stage":
            5,

        "block":
            "5.8",

        "phase":
            "pre_formal_acceptance_freeze",

        "status":
            "PASS_PREFORMAL_FREEZE",

        "formal_outcome_metrics_read":
            False,

        "formal_stage5_metrics_computed":
            False,

        "formal_results_used_for_parameter_selection":
            False,

        "post_hoc_tuning":
            False,

        "formal_manifest": {
            "path":
                str(
                    FORMAL_MANIFEST
                ),

            "sha256":
                manifest_sha,

            "scenario_count_contract":
                120,
        },

        "acceptance_policy": {
            "path":
                str(
                    POLICY
                ),

            "sha256":
                policy_sha,

            "coverage_method":
                FORMAL_COVERAGE_METHOD,

            "alpha":
                FORMAL_EMPIRICAL_ALPHA,

            "reported_q":
                list(
                    REQUESTED_COVERAGE_LEVELS
                ),

            "nominal_q":
                NOMINAL_COVERAGE_Q,
        },

        "next":
            (
                "Route-B one-scene schema probe, "
                "then deterministic pre-TTP materialization"
            ),
    }

    atomic_write(
        REPORT,
        canonical_bytes(
            report
        ),
    )

    print(
        "============================================================"
    )
    print(
        "BLOCK 5.8 PART 2/2 — PRE-FORMAL ACCEPTANCE FREEZE"
    )
    print(
        "============================================================"
    )
    print(
        "Part1 continuity          = PASS"
    )
    print(
        "formal outcomes read      = NO"
    )
    print(
        "formal Stage5 metrics     = NOT COMPUTED"
    )
    print(
        "coverage nominal q        = 0.95 FROZEN"
    )
    print(
        "coverage reported q       = 0.90 / 0.95 / 0.975 / 0.99"
    )
    print(
        "finite-sample method      = EXACT BINOMIAL LOWER TAIL"
    )
    print(
        "alpha                     = 0.05 FROZEN"
    )
    print(
        "fixed %-point tolerance   = NONE"
    )
    print(
        "post-hoc tuning           = NO"
    )
    print(
        "policy SHA256             =",
        policy_sha,
    )
    print(
        "STATUS                    = PASS_PREFORMAL_FREEZE"
    )
    print(
        "report                    =",
        REPORT,
    )


if __name__ == "__main__":
    main()

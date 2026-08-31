from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(
    "/home/agni/waymo/iscai_stage3"
)

REFERENCE = (
    ROOT
    / "artifacts/block38g/"
      "reference_block38f_report.json"
)

REPEAT = (
    ROOT
    / "artifacts/block38g/"
      "repeat_block38f_report.json"
)

REFERENCE_SCENES = (
    ROOT
    / "artifacts/block38g/"
      "reference_formal_per_scenario.jsonl"
)

REPEAT_SCENES = (
    ROOT
    / "artifacts/block38g/"
      "repeat_formal_per_scenario.jsonl"
)

FORMAL_MANIFEST = (
    ROOT
    / "artifacts/block38e/"
      "formal_validation_120.jsonl"
)

REPORT = (
    ROOT
    / "reports/"
      "block38g_reproducibility_gate.json"
)

EXPECTED_FORMAL_SHA = (
    "2208e7287ddf6439fda4597c435a9cba"
    "1d1b9d0e4c4547bc5dd92e56e8124e46"
)


def sha256_file(path):
    digest = sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def load_json(path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def strip_runtime(value):
    if isinstance(value, dict):
        return {
            key: strip_runtime(item)
            for key, item
            in value.items()
            if "runtime" not in key.lower()
        }

    if isinstance(value, list):
        return [
            strip_runtime(item)
            for item in value
        ]

    return value


print(
    "===== Stage3 Block 3.8G "
    "reproducibility gate ====="
)


# ------------------------------------------------------------
# Frozen formal manifest
# ------------------------------------------------------------

formal_sha = sha256_file(
    FORMAL_MANIFEST
)

if formal_sha != EXPECTED_FORMAL_SHA:
    raise SystemExit(
        "FAIL: formal manifest SHA changed."
    )


reference = load_json(
    REFERENCE
)

repeat = load_json(
    REPEAT
)

if (
    reference["status"] != "PASS"
    or repeat["status"] != "PASS"
):
    raise SystemExit(
        "FAIL: one formal run is not PASS."
    )

if (
    reference["scenario_count"] != 120
    or repeat["scenario_count"] != 120
):
    raise SystemExit(
        "FAIL: formal scenario count changed."
    )

if (
    reference[
        "formal_manifest_sha256"
    ]
    !=
    EXPECTED_FORMAL_SHA
    or
    repeat[
        "formal_manifest_sha256"
    ]
    !=
    EXPECTED_FORMAL_SHA
):
    raise SystemExit(
        "FAIL: run/manifest mismatch."
    )


# ------------------------------------------------------------
# Deterministic run hash
# ------------------------------------------------------------

reference_run_sha = (
    reference[
        "deterministic"
    ]["run_sha256"]
)

repeat_run_sha = (
    repeat[
        "deterministic"
    ]["run_sha256"]
)

if (
    len(reference_run_sha) != 64
    or any(
        char not in "0123456789abcdef"
        for char in reference_run_sha
    )
):
    raise SystemExit(
        "FAIL: repaired reference run SHA is invalid."
    )

if (
    repeat_run_sha
    !=
    reference_run_sha
):
    raise SystemExit(
        "FAIL: repeat deterministic run "
        "SHA differs."
    )


# ------------------------------------------------------------
# Per-scenario deterministic hashes
# ------------------------------------------------------------

reference_hashes = (
    reference[
        "deterministic"
    ]["scenario_hashes"]
)

repeat_hashes = (
    repeat[
        "deterministic"
    ]["scenario_hashes"]
)

if len(reference_hashes) != 120:
    raise SystemExit(
        "FAIL: reference scenario hash count."
    )

if len(repeat_hashes) != 120:
    raise SystemExit(
        "FAIL: repeat scenario hash count."
    )

if repeat_hashes != reference_hashes:
    differing = [
        i + 1
        for i, (a, b)
        in enumerate(
            zip(
                reference_hashes,
                repeat_hashes,
            )
        )
        if a != b
    ]

    raise SystemExit(
        "FAIL: per-scenario deterministic "
        f"hashes differ at ranks "
        f"{differing[:20]}"
    )


# ------------------------------------------------------------
# Exact metric equality, excluding runtime
# ------------------------------------------------------------

reference_primary = strip_runtime(
    reference[
        "primary_evaluation"
    ]
)

repeat_primary = strip_runtime(
    repeat[
        "primary_evaluation"
    ]
)

if (
    reference_primary
    !=
    repeat_primary
):
    raise SystemExit(
        "FAIL: primary formal metrics "
        "are not exactly reproducible."
    )


reference_supp = strip_runtime(
    reference[
        "supplementary_evaluation"
    ]
)

repeat_supp = strip_runtime(
    repeat[
        "supplementary_evaluation"
    ]
)

if reference_supp != repeat_supp:
    raise SystemExit(
        "FAIL: supplementary metrics "
        "are not exactly reproducible."
    )


# ------------------------------------------------------------
# Per-scenario artifact consistency
# ------------------------------------------------------------

reference_rows = load_jsonl(
    REFERENCE_SCENES
)

repeat_rows = load_jsonl(
    REPEAT_SCENES
)

if (
    len(reference_rows) != 120
    or len(repeat_rows) != 120
):
    raise SystemExit(
        "FAIL: per-scenario artifact count."
    )

reference_ids = [
    row["scenario_id"]
    for row in reference_rows
]

repeat_ids = [
    row["scenario_id"]
    for row in repeat_rows
]

if reference_ids != repeat_ids:
    raise SystemExit(
        "FAIL: scenario order differs."
    )

for rank, (first, second) in enumerate(
    zip(
        reference_rows,
        repeat_rows,
    ),
    start=1,
):
    if (
        first[
            "deterministic_sha256"
        ]
        !=
        second[
            "deterministic_sha256"
        ]
    ):
        raise SystemExit(
            "FAIL: scenario deterministic "
            f"artifact mismatch at rank "
            f"{rank}."
        )


# Raw JSONL file hashes are NOT required to match because
# measured wall-clock runtimes are intentionally included
# in those artifacts.

reference_raw_sha = sha256_file(
    REFERENCE_SCENES
)

repeat_raw_sha = sha256_file(
    REPEAT_SCENES
)


report = {
    "stage": 3,
    "block": "3.8G",
    "status": "PASS",

    "formal_manifest": {
        "scenario_count": 120,
        "sha256": formal_sha,
    },

    "reference": {
        "run_sha256":
            reference_run_sha,

        "raw_per_scenario_sha256":
            reference_raw_sha,
    },

    "repeat": {
        "run_sha256":
            repeat_run_sha,

        "raw_per_scenario_sha256":
            repeat_raw_sha,
    },

    "reproducibility": {
        "run_sha_exact":
            True,

        "scenario_hashes_exact":
            True,

        "scenario_hash_count":
            120,

        "primary_metrics_exact":
            True,

        "supplementary_metrics_exact":
            True,

        "scenario_order_exact":
            True,

        "raw_runtime_fields_required_exact":
            False,
    },

    "future_truth_algorithm_input":
        False,

    "formal_selection_changed":
        False,

    "models_or_thresholds_changed":
        False,
}


REPORT.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
    +
    "\n",
    encoding="utf-8",
)


print(
    "formal manifest SHA      = PASS"
)

print(
    "formal scenarios         = 120"
)

print(
    "reference run SHA        =",
    reference_run_sha,
)

print(
    "repeat run SHA           =",
    repeat_run_sha,
)

print(
    "run SHA exact            = PASS"
)

print(
    "120 scenario hashes      = PASS"
)

print(
    "primary metrics exact    = PASS"
)

print(
    "supplementary exact      = PASS"
)

print(
    "scenario order exact     = PASS"
)

print(
    "runtime equality required= NO"
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    REPORT,
)

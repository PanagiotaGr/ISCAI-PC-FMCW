from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Any


ROOT = Path("/home/agni/waymo")
S5 = ROOT / "iscai_stage5"

BLOCK58_CLOSURE = (
    S5
    / "artifacts/block58/block58_final_closure.json"
)

FORMAL_REPORT = (
    S5
    / "reports/block58_formal_evaluation.json"
)

FORMAL_RECORDS = (
    S5
    / "artifacts/block58/formal_clean_records.jsonl"
)

FORMAL_CACHE = (
    S5
    / "artifacts/block58/formal_clean_cache"
)

WORKER = (
    S5
    / "scripts/run_block59_repro_worker.py"
)

OUTPUT_A = (
    S5
    / "artifacts/block59/fresh_process_A.json"
)

OUTPUT_B = (
    S5
    / "artifacts/block59/fresh_process_B.json"
)

REPORT = (
    S5
    / "reports/block59_reproducibility.json"
)

MANIFEST = (
    S5
    / "artifacts/block59/block59_reproducibility_manifest.json"
)

CLOSURE = (
    S5
    / "artifacts/block59/block59_final_closure.json"
)

FAILURE = (
    S5
    / "reports/block59_failure.json"
)

EXPECTED_BLOCK58_CLOSURE_SHA = (
    "23ba634c421d264af245d5aa0013e380"
    "fd62f7b1b2e96fa1434162c2dad5038a"
)

EXPECTED_FORMAL_REPORT_SHA = (
    "76c23c21e667cf5acec1a6b1273256b9"
    "15f8c12d0bba6417459e56c27203fc55"
)

EXPECTED_FORMAL_RECORDS_SHA = (
    "b45c23d407bce2a10cb74db03b856773"
    "171470edaaa8770c2f032e974081800a"
)

EXPECTED_SELECTED_CACHE_SHA = {
    1:
        (
            "086d7d77a70af4c83db9f0ce728f881c"
            "fb907e8835643c7c9e366fe62bcfae49"
        ),

    60:
        (
            "bb228d436f3c53980c0d79bac37577e8"
            "f46f7c250c88518192940a62da85d800"
        ),

    120:
        (
            "17b9d4827d07b322290fa5720966c19f"
            "1aa1cdbf6bc3e89c603d0d80c2f4e289"
        ),
}

TARGET_RANKS = (
    1,
    60,
    120,
)

MIN_FREE_BYTES = (
    250
    *
    1024**3
)


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(
            message
        )


def sha256_file(
    path: Path,
) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(
                chunk
            )

    return h.hexdigest()


def canonical_bytes(
    value: Any,
) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        +
        "\n"
    ).encode(
        "utf-8"
    )


def canonical_sha(
    value: Any,
) -> str:
    return hashlib.sha256(
        canonical_bytes(
            value
        )
    ).hexdigest()


def atomic_json(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name
        +
        f".tmp.{os.getpid()}"
    )

    try:
        with tmp.open(
            "wb"
        ) as f:
            f.write(
                canonical_bytes(
                    payload
                )
            )

            f.flush()
            os.fsync(
                f.fileno()
            )

        os.replace(
            tmp,
            path,
        )

    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_once(
    path: Path,
    payload: Any,
) -> str:
    encoded = canonical_bytes(
        payload
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        existing = (
            path.read_bytes()
        )

        if existing == encoded:
            return (
                "ALREADY_IDENTICAL"
            )

        raise RuntimeError(
            "Existing frozen Block5.9 artifact "
            f"differs; refusing overwrite: {path}"
        )

    tmp = path.with_name(
        path.name
        +
        f".tmp.{os.getpid()}"
    )

    try:
        with tmp.open(
            "wb"
        ) as f:
            f.write(
                encoded
            )

            f.flush()
            os.fsync(
                f.fileno()
            )

        os.replace(
            tmp,
            path,
        )

    finally:
        if tmp.exists():
            tmp.unlink()

    return "CREATED"


def load_json(
    path: Path,
) -> dict[str, Any]:
    require(
        path.is_file(),
        f"Missing required JSON: {path}",
    )

    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    require(
        isinstance(
            value,
            dict,
        ),
        f"JSON root is not object: {path}",
    )

    return value


def cache_integrity() -> dict[str, Any]:
    require(
        FORMAL_CACHE.is_dir(),
        "Frozen formal_clean_cache is missing.",
    )

    files = sorted(
        FORMAL_CACHE.glob(
            "*.json"
        )
    )

    require(
        len(
            files
        )
        ==
        120,
        (
            "Frozen Block5.8 cache must contain "
            f"120 files; found {len(files)}."
        ),
    )

    combined = hashlib.sha256()

    selected = {}

    for index, path in enumerate(
        files,
        start=1,
    ):
        require(
            path.name.startswith(
                f"{index:03d}_"
            ),
            (
                "Formal cache ordering mismatch "
                f"at position {index}: {path.name}"
            ),
        )

        digest = (
            sha256_file(
                path
            )
        )

        combined.update(
            path.name.encode(
                "utf-8"
            )
        )
        combined.update(
            b"\0"
        )
        combined.update(
            digest.encode(
                "ascii"
            )
        )
        combined.update(
            b"\n"
        )

        if index in TARGET_RANKS:
            selected[
                index
            ] = {
                "path":
                    str(
                        path
                    ),

                "sha256":
                    digest,
            }

    for rank, expected in (
        EXPECTED_SELECTED_CACHE_SHA.items()
    ):
        require(
            selected[
                rank
            ][
                "sha256"
            ]
            ==
            expected,
            (
                "Frozen selected Block5.8 cache changed "
                f"for rank {rank}."
            ),
        )

    return {
        "count":
            120,

        "combined_manifest_sha256":
            combined.hexdigest(),

        "selected":
            selected,
    }


def file_map(
    paths: list[Path],
) -> dict[str, str]:
    result = {}

    for path in sorted(
        paths,
        key=lambda p:
            str(
                p
            ),
    ):
        if path.is_file():
            result[
                str(
                    path
                    .relative_to(
                        S5
                    )
                )
            ] = (
                sha256_file(
                    path
                )
            )

    return result


def implementation_fingerprint() -> dict[str, Any]:
    files = []

    files.extend(
        (
            S5
            /
            "src"
        ).rglob(
            "*.py"
        )
    )

    files.extend(
        (
            S5
            /
            "scripts"
        ).glob(
            "*.py"
        )
    )

    files.extend(
        (
            S5
            /
            "tests"
        ).glob(
            "*.py"
        )
    )

    files.extend(
        (
            S5
            /
            "configs"
        ).glob(
            "*.json"
        )
    )

    mapping = file_map(
        list(
            files
        )
    )

    h = hashlib.sha256()

    for name, digest in sorted(
        mapping.items()
    ):
        h.update(
            name.encode(
                "utf-8"
            )
        )
        h.update(
            b"\0"
        )
        h.update(
            digest.encode(
                "ascii"
            )
        )
        h.update(
            b"\n"
        )

    return {
        "file_count":
            len(
                mapping
            ),

        "sha256":
            h.hexdigest(),

        "files":
            mapping,
    }


def run_worker(
    label: str,
    output: Path,
) -> str:
    work_parent = (
        S5
        /
        "artifacts/block59"
    )

    work_parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    workdir = Path(
        tempfile.mkdtemp(
            prefix=(
                f".fresh_{label}_"
            ),
            dir=str(
                work_parent
            ),
        )
    )

    # Worker requires a path that does not yet exist.
    shutil.rmtree(
        workdir
    )

    command = [
        sys.executable,
        str(
            WORKER
        ),
        "--label",
        label,
        "--output",
        str(
            output
        ),
        "--workdir",
        str(
            workdir
        ),
    ]

    try:
        completed = (
            subprocess.run(
                command,
                cwd=str(
                    S5
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=1800,
                check=False,
            )
        )

        print(
            completed.stdout,
            end=(
                ""
                if completed.stdout.endswith(
                    "\n"
                )
                else
                "\n"
            ),
        )

        require(
            completed.returncode
            ==
            0,
            (
                f"Fresh process {label} failed "
                f"with exit code {completed.returncode}."
            ),
        )

        require(
            output.is_file(),
            (
                f"Fresh process {label} did not "
                "write its output."
            ),
        )

        return completed.stdout

    finally:
        if workdir.exists():
            shutil.rmtree(
                workdir
            )


def run_targeted_test() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        str(
            S5
            /
            "tests"
        ),
        "-p",
        "test_block59_reproducibility.py",
        "-v",
    ]

    completed = (
        subprocess.run(
            command,
            cwd=str(
                S5
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=300,
            check=False,
        )
    )

    print(
        completed.stdout,
        end=(
            ""
            if completed.stdout.endswith(
                "\n"
            )
            else
            "\n"
        ),
    )

    require(
        completed.returncode
        ==
        0,
        "Block5.9 targeted regression test failed.",
    )

    return {
        "status":
            "PASS",

        "command":
            " ".join(
                command
            ),

        "output_tail":
            completed.stdout[
                -6000:
            ],
    }


def success_already_frozen() -> bool:
    if not CLOSURE.is_file():
        return False

    closure = load_json(
        CLOSURE
    )

    require(
        closure.get(
            "status"
        )
        ==
        "PASS_FROZEN",
        (
            "Existing Block5.9 closure exists "
            "but is not PASS_FROZEN."
        ),
    )

    require(
        REPORT.is_file(),
        (
            "Block5.9 closure exists but "
            "final report is missing."
        ),
    )

    expected_report_sha = (
        closure[
            "provenance"
        ][
            "reproducibility_report_sha256"
        ]
    )

    require(
        sha256_file(
            REPORT
        )
        ==
        expected_report_sha,
        (
            "Existing frozen Block5.9 report "
            "changed after closure."
        ),
    )

    print(
        "BLOCK 5.9 already PASS / FROZEN."
    )
    print(
        "closure SHA256           =",
        sha256_file(
            CLOSURE
        ),
    )
    print(
        "NEXT                     = Block 5.10"
    )

    return True


def main() -> None:
    if success_already_frozen():
        return

    free = (
        shutil.disk_usage(
            S5
        ).free
    )

    require(
        free
        >=
        MIN_FREE_BYTES,
        (
            "Free storage below frozen 250 GiB reserve: "
            f"{free / 1024**3:.2f} GiB"
        ),
    )

    require(
        sha256_file(
            BLOCK58_CLOSURE
        )
        ==
        EXPECTED_BLOCK58_CLOSURE_SHA,
        "Frozen Block5.8 closure SHA changed.",
    )

    require(
        sha256_file(
            FORMAL_REPORT
        )
        ==
        EXPECTED_FORMAL_REPORT_SHA,
        "Frozen Block5.8 formal report SHA changed.",
    )

    require(
        sha256_file(
            FORMAL_RECORDS
        )
        ==
        EXPECTED_FORMAL_RECORDS_SHA,
        "Frozen Block5.8 formal records SHA changed.",
    )

    closure58 = load_json(
        BLOCK58_CLOSURE
    )

    require(
        closure58.get(
            "status"
        )
        ==
        "PASS_FROZEN",
        "Block5.8 is not PASS_FROZEN.",
    )

    formal = load_json(
        FORMAL_REPORT
    )

    require(
        formal.get(
            "status"
        )
        ==
        "PASS",
        "Frozen formal report is not PASS.",
    )

    require(
        formal[
            "formal_population"
        ][
            "N"
        ]
        ==
        120,
        "Frozen formal population is not N=120.",
    )

    before_cache = (
        cache_integrity()
    )

    before = {
        "block58_closure_sha256":
            sha256_file(
                BLOCK58_CLOSURE
            ),

        "formal_report_sha256":
            sha256_file(
                FORMAL_REPORT
            ),

        "formal_records_sha256":
            sha256_file(
                FORMAL_RECORDS
            ),

        "formal_cache":
            before_cache,
    }

    print(
        "="
        *
        72
    )
    print(
        "BLOCK 5.9 PART 2/2 — FRESH-PROCESS EXACT REPRODUCIBILITY"
    )
    print(
        "="
        *
        72
    )
    print(
        "Block5.8 closure         = SHA PASS"
    )
    print(
        "formal N=120             = FROZEN"
    )
    print(
        "repeat ranks             = 1 / 60 / 120"
    )
    print(
        "performance selection    = NO"
    )
    print(
        "fresh process A          = START"
    )

    run_worker(
        "A",
        OUTPUT_A,
    )

    print(
        "fresh process B          = START"
    )

    run_worker(
        "B",
        OUTPUT_B,
    )

    a = load_json(
        OUTPUT_A
    )

    b = load_json(
        OUTPUT_B
    )

    require(
        a.get(
            "status"
        )
        ==
        "PASS",
        "Fresh process A is not PASS.",
    )

    require(
        b.get(
            "status"
        )
        ==
        "PASS",
        "Fresh process B is not PASS.",
    )

    require(
        a[
            "canonical_runner"
        ][
            "sha256"
        ]
        ==
        formal[
            "provenance"
        ][
            "runner_sha256"
        ],
        (
            "Fresh process A did not use "
            "the formal report's canonical runner."
        ),
    )

    require(
        b[
            "canonical_runner"
        ][
            "sha256"
        ]
        ==
        formal[
            "provenance"
        ][
            "runner_sha256"
        ],
        (
            "Fresh process B did not use "
            "the formal report's canonical runner."
        ),
    )

    require(
        a[
            "all_fresh_semantics_equal_frozen"
        ]
        is True,
        "Fresh process A differs from frozen Block5.8 semantics.",
    )

    require(
        b[
            "all_fresh_semantics_equal_frozen"
        ]
        is True,
        "Fresh process B differs from frozen Block5.8 semantics.",
    )

    require(
        a[
            "semantic_signature_sha256"
        ]
        ==
        b[
            "semantic_signature_sha256"
        ],
        "Fresh-process semantic signatures differ A vs B.",
    )

    require(
        a[
            "trace_signature_sha256"
        ]
        ==
        b[
            "trace_signature_sha256"
        ],
        "Fresh-process deterministic trace differs A vs B.",
    )

    require(
        len(
            a[
                "scenes"
            ]
        )
        ==
        3,
        "Fresh process A scene count != 3.",
    )

    require(
        len(
            b[
                "scenes"
            ]
        )
        ==
        3,
        "Fresh process B scene count != 3.",
    )

    for scene_a, scene_b in zip(
        a[
            "scenes"
        ],
        b[
            "scenes"
        ],
    ):
        require(
            scene_a[
                "rank"
            ]
            ==
            scene_b[
                "rank"
            ],
            "Fresh process scene ranks differ.",
        )

        require(
            scene_a[
                "fresh_semantic_sha256"
            ]
            ==
            scene_b[
                "fresh_semantic_sha256"
            ],
            (
                "Fresh process semantic output differs "
                f"at rank {scene_a['rank']}."
            ),
        )

        require(
            scene_a[
                "trace_sha256"
            ]
            ==
            scene_b[
                "trace_sha256"
            ],
            (
                "Fresh process deterministic trace differs "
                f"at rank {scene_a['rank']}."
            ),
        )

    for category in (
        "receiver_posterior",
        "beam_probability",
        "topk_selection",
        "optical_metrics",
    ):
        require(
            a[
                "trace_category_counts"
            ][
                category
            ]
            >
            0,
            (
                "Missing required deterministic trace category: "
                f"{category}"
            ),
        )

        require(
            a[
                "trace_category_counts"
            ][
                category
            ]
            ==
            b[
                "trace_category_counts"
            ][
                category
            ],
            (
                "Trace call count differs A/B for "
                f"{category}."
            ),
        )

    require(
        a[
            "receiver_posterior_embedded_array_hash_count"
        ]
        >
        0,
        (
            "Receiver posterior exact numerical "
            "sample hashes were not captured."
        ),
    )

    require(
        a[
            "receiver_posterior_embedded_array_hash_count"
        ]
        ==
        b[
            "receiver_posterior_embedded_array_hash_count"
        ],
        "Receiver posterior numerical hash counts differ A/B.",
    )

    # Frozen Block5.8 artifacts must be byte-identical after
    # the two fresh processes.
    after_cache = (
        cache_integrity()
    )

    require(
        before_cache[
            "combined_manifest_sha256"
        ]
        ==
        after_cache[
            "combined_manifest_sha256"
        ],
        "Frozen Block5.8 cache mutated during Block5.9.",
    )

    require(
        sha256_file(
            BLOCK58_CLOSURE
        )
        ==
        before[
            "block58_closure_sha256"
        ],
        "Frozen Block5.8 closure mutated during Block5.9.",
    )

    require(
        sha256_file(
            FORMAL_REPORT
        )
        ==
        before[
            "formal_report_sha256"
        ],
        "Frozen formal report mutated during Block5.9.",
    )

    require(
        sha256_file(
            FORMAL_RECORDS
        )
        ==
        before[
            "formal_records_sha256"
        ],
        "Frozen formal records mutated during Block5.9.",
    )

    implementation = (
        implementation_fingerprint()
    )

    config_hashes = file_map(
        list(
            (
                S5
                /
                "configs"
            ).glob(
                "*.json"
            )
        )
    )

    optical_link = (
        S5
        /
        "src/iscai_stage5/optical_link.py"
    )

    candidate_report = {
        "stage":
            5,

        "block":
            "5.9",

        "phase":
            "exact_reproducibility_freeze",

        "status":
            "PASS_REPRODUCIBILITY_PRE_FREEZE",

        "predetermined_ranks":
            list(
                TARGET_RANKS
            ),

        "selection_performance_based":
            False,

        "fresh_process_count":
            2,

        "exact_match_to_frozen_block58":
            True,

        "fresh_process_repeat_exact":
            True,

        "receiver_posterior_samples_or_tensor_hashes_exact":
            True,

        "beam_probability_vectors_exact":
            True,

        "selected_topk_decisions_exact":
            True,

        "optical_metrics_exact":
            True,

        "block58_mutated":
            False,

        "canonical_runner": {
            "path":
                a[
                    "canonical_runner"
                ][
                    "path"
                ],

            "sha256":
                a[
                    "canonical_runner"
                ][
                    "sha256"
                ],

            "resolved_from_formal_report_provenance":
                True,
        },

        "fresh_process_signatures": {
            "semantic_sha256":
                a[
                    "semantic_signature_sha256"
                ],

            "trace_sha256":
                a[
                    "trace_signature_sha256"
                ],
        },

        "trace_contract": {
            "receiver_posterior_call_count":
                a[
                    "trace_category_counts"
                ][
                    "receiver_posterior"
                ],

            "beam_probability_call_count":
                a[
                    "trace_category_counts"
                ][
                    "beam_probability"
                ],

            "topk_selection_call_count":
                a[
                    "trace_category_counts"
                ][
                    "topk_selection"
                ],

            "optical_metrics_call_count":
                a[
                    "trace_category_counts"
                ][
                    "optical_metrics"
                ],

            "receiver_posterior_embedded_array_hash_count":
                a[
                    "receiver_posterior_embedded_array_hash_count"
                ],
        },

        "scene_repeats": [
            {
                "rank":
                    scene[
                        "rank"
                    ],

                "scenario_id":
                    scene[
                        "scenario_id"
                    ],

                "selected_prediction_available":
                    scene[
                        "selected_prediction_available"
                    ],

                "frozen_semantic_sha256":
                    scene[
                        "frozen_semantic_sha256"
                    ],

                "fresh_semantic_sha256":
                    scene[
                        "fresh_semantic_sha256"
                    ],

                "trace_sha256":
                    scene[
                        "trace_sha256"
                    ],

                "exact":
                    True,
            }
            for scene
            in a[
                "scenes"
            ]
        ],

        "frozen_block58": {
            "closure_sha256":
                before[
                    "block58_closure_sha256"
                ],

            "formal_report_sha256":
                before[
                    "formal_report_sha256"
                ],

            "formal_records_sha256":
                before[
                    "formal_records_sha256"
                ],

            "formal_cache_count":
                120,

            "formal_cache_combined_manifest_sha256":
                before_cache[
                    "combined_manifest_sha256"
                ],
        },

        "upstream_provenance":
            formal[
                "provenance"
            ],

        "stage5_config_sha256":
            config_hashes,

        "optical_link_adapter_sha256":
            sha256_file(
                optical_link
            ),

        "implementation":
            implementation,

        "worker_outputs": {
            "A": {
                "path":
                    str(
                        OUTPUT_A
                    ),

                "sha256":
                    sha256_file(
                        OUTPUT_A
                    ),
            },

            "B": {
                "path":
                    str(
                        OUTPUT_B
                    ),

                "sha256":
                    sha256_file(
                        OUTPUT_B
                    ),
            },
        },

        "next":
            "Run targeted Block5.9 regression, then freeze.",
    }

    atomic_json(
        REPORT,
        candidate_report,
    )

    test_evidence = (
        run_targeted_test()
    )

    final_report = dict(
        candidate_report
    )

    final_report[
        "status"
    ] = "PASS_FROZEN"

    final_report[
        "targeted_regression"
    ] = test_evidence

    final_report[
        "next"
    ] = "Block 5.10 final Stage5 closure and Stage5→Stage6 handoff"

    atomic_json(
        REPORT,
        final_report,
    )

    final_report_sha = (
        sha256_file(
            REPORT
        )
    )

    manifest_payload = {
        "stage":
            5,

        "block":
            "5.9",

        "status":
            "PASS_FROZEN",

        "reproducibility_report": {
            "path":
                str(
                    REPORT
                ),

            "sha256":
                final_report_sha,
        },

        "fresh_process_A": {
            "path":
                str(
                    OUTPUT_A
                ),

            "sha256":
                sha256_file(
                    OUTPUT_A
                ),
        },

        "fresh_process_B": {
            "path":
                str(
                    OUTPUT_B
                ),

            "sha256":
                sha256_file(
                    OUTPUT_B
                ),
        },

        "block58_closure_sha256":
            EXPECTED_BLOCK58_CLOSURE_SHA,

        "formal_report_sha256":
            EXPECTED_FORMAL_REPORT_SHA,

        "formal_records_sha256":
            EXPECTED_FORMAL_RECORDS_SHA,

        "formal_cache_combined_manifest_sha256":
            after_cache[
                "combined_manifest_sha256"
            ],

        "canonical_runner_sha256":
            formal[
                "provenance"
            ][
                "runner_sha256"
            ],

        "implementation_sha256":
            implementation[
                "sha256"
            ],

        "implementation_file_count":
            implementation[
                "file_count"
            ],

        "all_stage5_configs":
            config_hashes,

        "optical_link_adapter_sha256":
            sha256_file(
                optical_link
            ),

        "deterministic_repeat": {
            "ranks":
                list(
                    TARGET_RANKS
                ),

            "semantic_sha256":
                a[
                    "semantic_signature_sha256"
                ],

            "trace_sha256":
                a[
                    "trace_signature_sha256"
                ],
        },
    }

    atomic_json(
        MANIFEST,
        manifest_payload,
    )

    manifest_sha = (
        sha256_file(
            MANIFEST
        )
    )

    closure_payload = {
        "stage":
            5,

        "block":
            "5.9",

        "phase":
            "reproducibility_closure",

        "status":
            "PASS_FROZEN",

        "block58_remained_frozen":
            True,

        "fresh_process_repeat":
            "PASS_EXACT",

        "frozen_semantic_reconstruction":
            "PASS_EXACT",

        "receiver_posterior_samples_hash":
            "PASS_EXACT",

        "beam_probability_vector":
            "PASS_EXACT",

        "selected_topk_set":
            "PASS_EXACT",

        "optical_metrics":
            "PASS_EXACT",

        "targeted_regression":
            "PASS",

        "provenance": {
            "block58_closure_sha256":
                EXPECTED_BLOCK58_CLOSURE_SHA,

            "reproducibility_report_sha256":
                final_report_sha,

            "reproducibility_manifest_sha256":
                manifest_sha,

            "fresh_process_A_sha256":
                sha256_file(
                    OUTPUT_A
                ),

            "fresh_process_B_sha256":
                sha256_file(
                    OUTPUT_B
                ),

            "canonical_runner_sha256":
                formal[
                    "provenance"
                ][
                    "runner_sha256"
                ],

            "implementation_sha256":
                implementation[
                    "sha256"
                ],
        },

        "next":
            "Block 5.10 final Stage5 closure and Stage5→Stage6 handoff",
    }

    closure_write = (
        atomic_write_once(
            CLOSURE,
            closure_payload,
        )
    )

    closure_sha = (
        sha256_file(
            CLOSURE
        )
    )

    print(
        "="
        *
        72
    )
    print(
        "BLOCK 5.9 REPRODUCIBILITY RESULT"
    )
    print(
        "="
        *
        72
    )
    print(
        "fresh process A          = PASS"
    )
    print(
        "fresh process B          = PASS"
    )
    print(
        "frozen semantic repeat   = EXACT"
    )
    print(
        "A/B semantic repeat      = EXACT"
    )
    print(
        "receiver posterior hash  = EXACT"
    )
    print(
        "beam probability vector  = EXACT"
    )
    print(
        "selected Top-K           = EXACT"
    )
    print(
        "optical metrics          = EXACT"
    )
    print(
        "Block5.8 mutation        = NO"
    )
    print(
        "targeted regression      = PASS"
    )
    print(
        "implementation SHA256    =",
        implementation[
            "sha256"
        ],
    )
    print(
        "manifest SHA256          =",
        manifest_sha,
    )
    print(
        "closure write            =",
        closure_write,
    )
    print(
        "closure SHA256           =",
        closure_sha,
    )
    # A previous failed attempt may have left a BLOCKED diagnostic.
    # Once the current Block5.9 run has fully passed and frozen, that
    # stale failure report must not coexist with the canonical PASS.
    if FAILURE.is_file():
        FAILURE.unlink()

    print(
        "BLOCK 5.9 STATUS         = PASS / FROZEN"
    )
    print(
        "NEXT                     = Block 5.10"
    )


def guarded_main() -> None:
    try:
        main()

    except Exception as exc:
        payload = {
            "stage":
                5,

            "block":
                "5.9",

            "status":
                "BLOCKED",

            "exception_type":
                type(
                    exc
                ).__name__,

            "message":
                str(
                    exc
                ),

            "traceback":
                traceback.format_exc(),

            "recovery":
                (
                    "Do not modify frozen Block5.8 or Stage4. "
                    "Send the complete Block5.9 terminal output "
                    "and failure report; repair Block5.9 only."
                ),
        }

        try:
            atomic_json(
                FAILURE,
                payload,
            )

        except Exception:
            pass

        raise


if __name__ == "__main__":
    guarded_main()

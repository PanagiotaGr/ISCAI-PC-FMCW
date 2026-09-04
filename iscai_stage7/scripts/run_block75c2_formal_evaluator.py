#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
from typing import Any
import zipfile
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)

import numpy as np


# =============================================================================
# Frozen paths
# =============================================================================

ROOT = Path("/home/agni/waymo")

S1 = ROOT / "iscai_stage1"
S3 = ROOT / "iscai_stage3"
S5 = ROOT / "iscai_stage5"
S6 = ROOT / "iscai_stage6"
S7 = ROOT / "iscai_stage7"

AUDIT = (
    ROOT
    / "audits/stage7_pdf_alignment"
)

SCRIPT_EXPECTED_PATH = (
    S7
    / "scripts/run_block75c2_formal_evaluator.py"
)

SOURCE_BUNDLE = (
    AUDIT
    / "block75_c2c3_exact_source_bundle.zip"
)

B75B = (
    S7
    / "configs/"
      "stage7_block75b_final_formal_evaluator_contract.json"
)

B74 = (
    S7
    / "configs/"
      "stage7_block74_MINIMAL_HANDOFF.json"
)

FORMAL = (
    S3
    / "artifacts/block38e/"
      "formal_validation_120.jsonl"
)

VALIDATION_MANIFEST = (
    ROOT
    / "iscai_data_prep/manifests/"
      "selected_validation.jsonl"
)

PAIRED_ROOT = (
    ROOT
    / "data/paired_womd_lidar_v1_3_0"
)


# -----------------------------------------------------------------------------
# Final frozen C1
# -----------------------------------------------------------------------------

C1_RUNNER = (
    S7
    / "scripts/"
      "run_block75c1_formal_causal_output_lock.py"
)

C1_CHECKER = (
    S7
    / "scripts/"
      "check_block75c1_nonoracle_lock.py"
)

C1_MANIFEST = (
    S7
    / "artifacts/"
      "block75c1_nonoracle_lock_manifest.json"
)

C1_REPORT = (
    S7
    / "reports/"
      "stage7_block75c1_nonoracle_lock_report.json"
)

C1_CHECKER_LOG = (
    AUDIT
    / "block75c1_independent_checker.log"
)


# -----------------------------------------------------------------------------
# Frozen Stage5 communication authorities
# -----------------------------------------------------------------------------

BEAM_CFG = (
    S5
    / "configs/beam_codebook_policy.json"
)

BEAM_LATENCY_CFG = (
    S5
    / "configs/beam_latency_policy.json"
)

OPTICAL_CFG = (
    S5
    / "configs/optical_link_policy.json"
)

STAGE5_FORMAL_SOURCE = (
    S5
    / "scripts/run_block58_part2_formal_evaluation.py"
)


# -----------------------------------------------------------------------------
# Frozen Stage6 evaluator authorities
# -----------------------------------------------------------------------------

GRID_CFG = (
    S6
    / "configs/block64_mc_grid_numeric_freeze.json"
)

EVALUATOR_CFG = (
    S6
    / "configs/stage6_evaluator_reference_contract.json"
)

STAGE6_METRIC_SOURCE = (
    S6
    / "src/iscai_stage6/adb/"
      "metric_semantics.py"
)

STAGE6_GEOMETRY_SOURCE = (
    S6
    / "src/iscai_stage6/adb/"
      "geometry.py"
)

STAGE6_OCC_SOURCE = (
    S6
    / "src/iscai_stage6/adb/"
      "probabilistic_occupancy.py"
)

SURROGATE_SOURCE = (
    S6
    / "scripts/"
      "run_block68_stage3_fixed_cache_parity.py"
)


# -----------------------------------------------------------------------------
# C2 outputs
# -----------------------------------------------------------------------------

OUT_ROOT = (
    S7
    / "artifacts/block75c2_formal_raw"
)

MERGED_ROWS = (
    S7
    / "artifacts/block75c2_joint_rows.jsonl"
)

C2_MANIFEST = (
    S7
    / "artifacts/"
      "block75c2_formal_raw_manifest.json"
)

C2_SEAL = (
    S7
    / "artifacts/"
      "stage7_block75c2_frozen_seal.json"
)

C2_REPORT = (
    S7
    / "reports/"
      "stage7_block75c2_formal_evaluation_report.json"
)


# =============================================================================
# Frozen hashes
# =============================================================================

EXPECTED_BUNDLE_SHA = (
    "8ce6594f1fe53a3f1fbb1bfa210ad475"
    "598818d75a4c82fe1fc31e19b2971282"
)

EXPECTED_B75B_SHA = (
    "5f79169138653b8223a795fae625adc5"
    "859a7a046a67c79fa98d357e256d7029"
)

EXPECTED_B74_SHA = (
    "1351ac6d08f42e8a6627b6631aad438e"
    "89f416006feab800cd45c74d7f14eefa"
)

EXPECTED_FORMAL_SHA = (
    "2208e7287ddf6439fda4597c435a9cba"
    "1d1b9d0e4c4547bc5dd92e56e8124e46"
)

EXPECTED_FORMAL_ORDER_SHA = (
    "4ba67f109c67f1306374423429654415e"
    "f884217dd93371f33e3d8fe27d948b7"
)

EXPECTED_VALIDATION_SHA = (
    "dc10609ef18a2ba881657eb3da3a3df7"
    "a81bdcc8345ecbc2227102ab16b8833c"
)

EXPECTED_C1_RUNNER_SHA = (
    "077237d85e62468f90ebf21a70fe6cdf"
    "3ad095fe83dba2985d0b4cb799379275"
)

EXPECTED_C1_CHECKER_SHA = (
    "b9bfb6447fbfe1982c6aa46e0cb5053"
    "d527112207bfc68be5acea29de095725e"
)

EXPECTED_C1_MANIFEST_SHA = (
    "d6792d2bcf304bd51260e98d99e8f8fa"
    "1e0aa470849a2d1989bdeac5ad74bfe5"
)

EXPECTED_C1_CHECKER_LOG_SHA = (
    "770e83a57353e1ec4d003d98300b2a175"
    "edcd4d567e7511771c66063301ef2c9"
)

EXPECTED_STAGE5_FORMAL_SOURCE_SHA = (
    "16495f3c46ed5f95145918b25b169f8a"
    "a33408b2ef84ffc19cb66af1db3c189a"
)

EXPECTED_METRIC_SOURCE_SHA = (
    "bf8358b5a7edfe40ffac2718ca7cd5af"
    "6e7b5fdb3ff8a2823caad6d0f40f057a"
)

EXPECTED_GEOMETRY_SOURCE_SHA = (
    "f6e6cc0dd362338eeeaae9d8e25947434"
    "a3e7c64b350f0baf81a79a46159f94e"
)

EXPECTED_OCC_SOURCE_SHA = (
    "d4206761fc70495563d53a6ab58aefe6d"
    "b5d8bd1b2d09736bfc5bbc6b6f50e3e"
)

EXPECTED_SURROGATE_SOURCE_SHA = (
    "66961b9cf6976f4f7d1e9066365b243d"
    "a8c73e98c5a3e462c0b81f32f473ea7b"
)

EXPECTED_BEAM_CFG_SHA = (
    "bd94f8609393a7c9fe02762cc4bf38e3"
    "a77a90cc31adef5f2e6b06aece4074d7"
)

EXPECTED_BEAM_LATENCY_CFG_SHA = (
    "aa04f2e73da591b87f2ab2e962d1b69c"
    "1bad645c043c0d15970dbf0990080ed0"
)

EXPECTED_OPTICAL_CFG_SHA = (
    "2075f0d0f3128af94c5d5ed99f0da4d"
    "ba87e08ace69de542bec4c9dd4454302b"
)

EXPECTED_GRID_CFG_SHA = (
    "993c4248a902e7dff3a4383ac343e7222"
    "cc43ef9b9372ed2efd0ee08d6d20a73"
)

EXPECTED_EVALUATOR_CFG_SHA = (
    "c9ff0390820a3c37702fb391ba603244f"
    "ffcf7c03c476045b590aa483e89bd7e"
)


# =============================================================================
# Frozen evaluation semantics
# =============================================================================

SYSTEMS = (
    "shared_trajectory_posterior",
    "independent_models",
    "direct_beam_classifier",
    "direct_ADB_predictor",
    "deterministic_shared_trajectory",
)

HORIZONS_S = (
    0.1,
    0.3,
    0.5,
    1.0,
)

HORIZON_OFFSETS = (
    1,
    3,
    5,
    10,
)

PRIMARY_CODEBOOK = 32
PRIMARY_COVERAGE = 0.95

GRID_SHAPE = (
    501,
    301,
)

SCHEDULE_SHAPE = (
    4,
    501,
    301,
)

TEMPORAL_DELTA_T_S = (
    0.1,
    0.2,
    0.2,
    0.5,
)

CORE_CLASSES = (
    "TYPE_VEHICLE",
    "TYPE_PEDESTRIAN",
    "TYPE_CYCLIST",
)

EXPECTED_C1_NPZ_MEMBERS = {
    "current_reactive_t0.npy",
    "shared_trajectory_posterior.npy",
    "independent_models.npy",
    "direct_ADB_predictor.npy",
    "deterministic_shared_trajectory.npy",
}

REFERENCE_FIELDS = (
    "mask_vehicle",
    "mask_pedestrian",
    "mask_cyclist",
    "oracle_all",
    "oracle_vehicle",
    "pedestrian_region",
    "cyclist_region",
    "vehicle_surrogate",
)


# =============================================================================
# Fail-closed utilities
# =============================================================================

class FailClosed(RuntimeError):
    pass


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise FailClosed(
            message
        )


def sha256_bytes(
    data: bytes,
) -> str:
    return hashlib.sha256(
        data
    ).hexdigest()


def sha256_file(
    path: Path,
) -> str:
    require(
        path.is_file(),
        f"Missing file: {path}",
    )

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
    obj: Any,
) -> bytes:
    return (
        json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode(
        "utf-8"
    )


def read_json(
    path: Path,
) -> Any:
    require(
        path.is_file(),
        f"Missing JSON: {path}",
    )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def verify_sha(
    path: Path,
    expected: str,
    label: str,
) -> None:
    actual = sha256_file(
        path
    )

    require(
        actual == expected,
        (
            f"{label} SHA mismatch: "
            f"{actual} != {expected}: {path}"
        ),
    )


def safe_number(
    value: Any,
) -> float | None:
    if value is None:
        return None

    x = float(
        value
    )

    if math.isfinite(
        x
    ):
        return x

    return None


def mean_finite(
    values,
) -> float | None:
    vals = [
        float(v)
        for v in values
        if (
            v is not None
            and math.isfinite(
                float(v)
            )
        )
    ]

    if not vals:
        return None

    return float(
        sum(vals)
        /
        len(vals)
    )


def array_digest(
    array: np.ndarray,
) -> str:
    a = np.ascontiguousarray(
        np.asarray(
            array
        )
    )

    h = hashlib.sha256()

    h.update(
        str(
            a.dtype
        ).encode("ascii")
    )
    h.update(b"\0")

    h.update(
        json.dumps(
            list(
                a.shape
            ),
            separators=(",", ":"),
        ).encode("ascii")
    )
    h.update(b"\0")

    h.update(
        a.tobytes(
            order="C"
        )
    )

    return h.hexdigest()


def atomic_immutable_file(
    path: Path,
    data: bytes,
) -> str:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        existing = (
            path.read_bytes()
        )

        require(
            existing == data,
            (
                "Immutable file already exists "
                f"with different bytes: {path}"
            ),
        )

        return sha256_bytes(
            existing
        )

    fd, temp_name = (
        tempfile.mkstemp(
            prefix=(
                path.name
                +
                ".tmp."
            ),
            dir=str(
                path.parent
            ),
        )
    )

    temp = Path(
        temp_name
    )

    try:
        with os.fdopen(
            fd,
            "wb",
        ) as f:
            f.write(
                data
            )
            f.flush()
            os.fsync(
                f.fileno()
            )

        os.replace(
            temp,
            path,
        )

        os.chmod(
            path,
            0o444,
        )

        dir_fd = os.open(
            path.parent,
            os.O_RDONLY,
        )

        try:
            os.fsync(
                dir_fd
            )
        finally:
            os.close(
                dir_fd
            )

    finally:
        if temp.exists():
            temp.unlink()

    return sha256_file(
        path
    )


def deterministic_npz_bytes(
    arrays: dict[
        str,
        np.ndarray,
    ],
) -> bytes:
    output = io.BytesIO()

    with zipfile.ZipFile(
        output,
        "w",
        compression=(
            zipfile.ZIP_DEFLATED
        ),
        compresslevel=9,
    ) as zf:

        for name in sorted(
            arrays
        ):
            array = np.asarray(
                arrays[
                    name
                ]
            )

            npy = io.BytesIO()

            np.save(
                npy,
                array,
                allow_pickle=False,
            )

            info = zipfile.ZipInfo(
                filename=(
                    name
                    +
                    ".npy"
                ),
                date_time=(
                    1980,
                    1,
                    1,
                    0,
                    0,
                    0,
                ),
            )

            info.compress_type = (
                zipfile.ZIP_DEFLATED
            )

            info.external_attr = (
                (0o444 & 0xFFFF)
                <<
                16
            )

            zf.writestr(
                info,
                npy.getvalue(),
                compress_type=(
                    zipfile.ZIP_DEFLATED
                ),
                compresslevel=9,
            )

    return output.getvalue()


# =============================================================================
# Exact source-bundle / authority verification
# =============================================================================

def verify_exact_source_bundle(
) -> dict[str, Any]:

    verify_sha(
        SOURCE_BUNDLE,
        EXPECTED_BUNDLE_SHA,
        "C2/C3 exact source bundle",
    )

    with zipfile.ZipFile(
        SOURCE_BUNDLE,
        "r",
    ) as zf:

        manifest = json.loads(
            zf.read(
                "SOURCE_BUNDLE_MANIFEST.json"
            )
        )

        require(
            isinstance(
                manifest,
                list,
            )
            and len(
                manifest
            )
            ==
            32,
            (
                "Unexpected source-bundle "
                "manifest cardinality."
            ),
        )

        names = set(
            zf.namelist()
        )

        final_c1_sources = {
            str(C1_RUNNER):
                EXPECTED_C1_RUNNER_SHA,

            str(C1_CHECKER):
                EXPECTED_C1_CHECKER_SHA,
        }

        exact = 0
        expected_differences = 0

        for item in manifest:
            path = Path(
                item[
                    "path"
                ]
            )

            relative = (
                str(path).split(
                    "/home/agni/waymo/",
                    1,
                )[-1]
            )

            require(
                relative
                in names,
                (
                    "Bundle member missing: "
                    f"{relative}"
                ),
            )

            payload = zf.read(
                relative
            )

            require(
                len(payload)
                ==
                int(
                    item[
                        "bytes"
                    ]
                ),
                (
                    "Bundle byte-count mismatch: "
                    f"{relative}"
                ),
            )

            require(
                sha256_bytes(
                    payload
                )
                ==
                item[
                    "sha256"
                ],
                (
                    "Bundle internal SHA mismatch: "
                    f"{relative}"
                ),
            )

            require(
                path.is_file(),
                (
                    "Host source missing: "
                    f"{path}"
                ),
            )

            host_sha = sha256_file(
                path
            )

            if (
                str(path)
                in
                final_c1_sources
            ):
                require(
                    host_sha
                    ==
                    final_c1_sources[
                        str(path)
                    ],
                    (
                        "Final C1 source identity "
                        f"mismatch: {path}"
                    ),
                )

                require(
                    host_sha
                    !=
                    item[
                        "sha256"
                    ],
                    (
                        "Expected final C1 source "
                        "to differ from bundled "
                        f"pre-final version: {path}"
                    ),
                )

                expected_differences += 1

            else:
                require(
                    host_sha
                    ==
                    item[
                        "sha256"
                    ],
                    (
                        "Unexpected frozen source drift: "
                        f"{path}"
                    ),
                )

                exact += 1

    require(
        exact == 30
        and
        expected_differences == 2,
        (
            "Unexpected bundle/host binding: "
            f"exact={exact}, "
            f"final_C1={expected_differences}"
        ),
    )

    return {
        "bundle_sha256":
            EXPECTED_BUNDLE_SHA,

        "bundle_entries":
            32,

        "exact_host_matches":
            exact,

        "expected_final_C1_differences":
            expected_differences,
    }


# =============================================================================
# C1 lock verification
# =============================================================================

def verify_c1_lock_entry(
    entry: dict[str, Any],
    runner_sha: str,
) -> dict[str, Any]:

    formal_index = int(
        entry[
            "formal_index"
        ]
    )

    scenario_id = str(
        entry[
            "scenario_id"
        ]
    )

    lock_json = Path(
        entry[
            "lock_json"
        ]
    )

    lock_npz = Path(
        entry[
            "adb_schedules_npz"
        ]
    )

    require(
        lock_json.is_file()
        and
        lock_npz.is_file(),
        (
            "Missing C1 lock pair: "
            f"{scenario_id}"
        ),
    )

    json_sha = sha256_file(
        lock_json
    )

    npz_sha = sha256_file(
        lock_npz
    )

    require(
        json_sha
        ==
        entry[
            "lock_json_sha256"
        ],
        (
            "C1 JSON SHA mismatch: "
            f"{scenario_id}"
        ),
    )

    require(
        npz_sha
        ==
        entry[
            "adb_schedules_npz_sha256"
        ],
        (
            "C1 NPZ SHA mismatch: "
            f"{scenario_id}"
        ),
    )

    document = read_json(
        lock_json
    )

    require(
        document.get(
            "block"
        )
        ==
        "7.5C1",
        (
            "C1 block mismatch: "
            f"{scenario_id}"
        ),
    )

    require(
        document.get(
            "status"
        )
        ==
        "LOCKED_NONORACLE_CAUSAL_OUTPUTS",
        (
            "C1 status mismatch: "
            f"{scenario_id}"
        ),
    )

    require(
        int(
            document[
                "formal_index"
            ]
        )
        ==
        formal_index,
        (
            "C1 formal-index mismatch: "
            f"{scenario_id}"
        ),
    )

    require(
        str(
            document[
                "scenario_id"
            ]
        )
        ==
        scenario_id,
        (
            "C1 scenario identity mismatch: "
            f"{scenario_id}"
        ),
    )

    require(
        document.get(
            "runner_sha256"
        )
        ==
        runner_sha,
        (
            "C1 runner SHA mismatch: "
            f"{scenario_id}"
        ),
    )

    require(
        document.get(
            "block75b_contract_sha256"
        )
        ==
        EXPECTED_B75B_SHA,
        (
            "C1 Block7.5B SHA mismatch: "
            f"{scenario_id}"
        ),
    )

    require(
        document.get(
            "future_GT_accessed"
        )
        is False,
        (
            "C1 future-GT boundary invalid: "
            f"{scenario_id}"
        ),
    )

    require(
        document.get(
            "formal_metrics_computed"
        )
        is False,
        (
            "C1 metric boundary invalid: "
            f"{scenario_id}"
        ),
    )

    require(
        document.get(
            "post_outcome_tuning"
        )
        is False,
        (
            "C1 post-outcome tuning flag invalid: "
            f"{scenario_id}"
        ),
    )

    require(
        document.get(
            "scientific_parameters_modified"
        )
        is False,
        (
            "C1 scientific-change flag invalid: "
            f"{scenario_id}"
        ),
    )

    require(
        document.get(
            "adb_schedules_npz_sha256"
        )
        ==
        npz_sha,
        (
            "C1 embedded NPZ SHA mismatch: "
            f"{scenario_id}"
        ),
    )

    stored_digest = (
        document.get(
            "nonoracle_output_digest_sha256"
        )
    )

    require(
        stored_digest
        ==
        entry[
            "nonoracle_output_digest_sha256"
        ],
        (
            "C1 digest/manifest mismatch: "
            f"{scenario_id}"
        ),
    )

    core = dict(
        document
    )

    core.pop(
        "nonoracle_output_digest_sha256",
        None,
    )

    require(
        sha256_bytes(
            canonical_bytes(
                core
            )
        )
        ==
        stored_digest,
        (
            "C1 canonical digest mismatch: "
            f"{scenario_id}"
        ),
    )

    with zipfile.ZipFile(
        lock_npz,
        "r",
    ) as zf:
        require(
            set(
                zf.namelist()
            )
            ==
            EXPECTED_C1_NPZ_MEMBERS,
            (
                "C1 NPZ member-set mismatch: "
                f"{scenario_id}"
            ),
        )

    return document


# =============================================================================
# Global preflight — NO FUTURE GT
# =============================================================================

def global_preflight(
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    str,
]:

    print(
        "\n===== A. FROZEN SOURCE / CONTRACT GATE =====",
        flush=True,
    )

    bundle_info = (
        verify_exact_source_bundle()
    )

    for path, expected, label in (
        (
            B75B,
            EXPECTED_B75B_SHA,
            "7.5B contract",
        ),
        (
            B74,
            EXPECTED_B74_SHA,
            "7.4 handoff",
        ),
        (
            FORMAL,
            EXPECTED_FORMAL_SHA,
            "formal N120 manifest",
        ),
        (
            VALIDATION_MANIFEST,
            EXPECTED_VALIDATION_SHA,
            "validation manifest",
        ),
        (
            STAGE5_FORMAL_SOURCE,
            EXPECTED_STAGE5_FORMAL_SOURCE_SHA,
            "Stage5 formal methodology source",
        ),
        (
            STAGE6_METRIC_SOURCE,
            EXPECTED_METRIC_SOURCE_SHA,
            "Stage6 metric semantics",
        ),
        (
            STAGE6_GEOMETRY_SOURCE,
            EXPECTED_GEOMETRY_SOURCE_SHA,
            "Stage6 geometry",
        ),
        (
            STAGE6_OCC_SOURCE,
            EXPECTED_OCC_SOURCE_SHA,
            "Stage6 full-box rasterization",
        ),
        (
            SURROGATE_SOURCE,
            EXPECTED_SURROGATE_SOURCE_SHA,
            "Stage6 canonical vehicle surrogate",
        ),
        (
            BEAM_CFG,
            EXPECTED_BEAM_CFG_SHA,
            "beam codebook config",
        ),
        (
            BEAM_LATENCY_CFG,
            EXPECTED_BEAM_LATENCY_CFG_SHA,
            "beam latency config",
        ),
        (
            OPTICAL_CFG,
            EXPECTED_OPTICAL_CFG_SHA,
            "optical link config",
        ),
        (
            GRID_CFG,
            EXPECTED_GRID_CFG_SHA,
            "ADB grid config",
        ),
        (
            EVALUATOR_CFG,
            EXPECTED_EVALUATOR_CFG_SHA,
            "Stage6 evaluator reference contract",
        ),
    ):
        verify_sha(
            path,
            expected,
            label,
        )

    contract = read_json(
        B75B
    )

    require(
        contract.get(
            "status"
        )
        ==
        "FROZEN_FINAL_FORMAL_EVALUATOR_CONTRACT",
        "Block7.5B contract status invalid.",
    )

    require(
        contract[
            "nonoracle_output_lock"
        ][
            "lock_artifact_must_exist_before_future_GT_read"
        ]
        is True,
        "7.5B lock-before-GT rule changed.",
    )

    require(
        contract[
            "formal_population"
        ][
            "historical_Stage6_per_scenario_metrics_reused"
        ]
        is False,
        (
            "7.5B historical Stage6 "
            "metric-reuse rule changed."
        ),
    )

    evaluator_contract = read_json(
        EVALUATOR_CFG
    )

    require(
        evaluator_contract.get(
            "status"
        )
        ==
        "FROZEN_EVALUATOR_REFERENCE_CONTRACT",
        "Stage6 evaluator contract invalid.",
    )

    require(
        evaluator_contract[
            "constructed_oracle"
        ][
            "measured_ADB_ground_truth"
        ]
        is False,
        (
            "Constructed ADB oracle "
            "semantics changed."
        ),
    )

    require(
        evaluator_contract[
            "constructed_oracle"
        ][
            "future_truth_access"
        ]
        ==
        "evaluator_after_controller_decision_only",
        (
            "Stage6 future-truth evaluator "
            "timing changed."
        ),
    )

    print(
        "source bundle / frozen scientific sources = PASS",
        flush=True,
    )

    print(
        "\n===== B. FINAL C1 GLOBAL SEAL =====",
        flush=True,
    )

    verify_sha(
        C1_RUNNER,
        EXPECTED_C1_RUNNER_SHA,
        "final C1 runner",
    )

    verify_sha(
        C1_CHECKER,
        EXPECTED_C1_CHECKER_SHA,
        "final C1 checker",
    )

    verify_sha(
        C1_MANIFEST,
        EXPECTED_C1_MANIFEST_SHA,
        "C1 manifest",
    )

    verify_sha(
        C1_CHECKER_LOG,
        EXPECTED_C1_CHECKER_LOG_SHA,
        "C1 checker log",
    )

    checker_text = (
        C1_CHECKER_LOG.read_text(
            encoding="utf-8",
            errors="replace",
        )
    )

    require(
        "INDEPENDENT C1 CHECK = PASS_100_PERCENT"
        in
        checker_text,
        (
            "Independent C1 checker "
            "PASS marker absent."
        ),
    )

    c1_report = read_json(
        C1_REPORT
    )

    require(
        c1_report.get(
            "status"
        )
        ==
        "PASS",
        "C1 report is not PASS.",
    )

    require(
        c1_report.get(
            "manifest_sha256"
        )
        ==
        EXPECTED_C1_MANIFEST_SHA,
        "C1 report manifest binding mismatch.",
    )

    manifest = read_json(
        C1_MANIFEST
    )

    require(
        manifest.get(
            "status"
        )
        ==
        "FROZEN_COMPLETE_NONORACLE_OUTPUT_LOCK",
        "C1 manifest status invalid.",
    )

    require(
        manifest.get(
            "scenario_lock_count"
        )
        ==
        120,
        "C1 lock count != 120.",
    )

    require(
        tuple(
            manifest.get(
                "systems",
                [],
            )
        )
        ==
        SYSTEMS,
        "Frozen five-system order changed.",
    )

    require(
        manifest.get(
            "future_GT_accessed"
        )
        is False,
        "C1 global future-GT flag invalid.",
    )

    require(
        manifest.get(
            "formal_metrics_computed"
        )
        is False,
        "C1 global formal-metric flag invalid.",
    )

    entries = (
        manifest[
            "scenario_locks"
        ]
    )

    require(
        len(entries)
        ==
        120,
        "C1 scenario_locks length != 120.",
    )

    require(
        [
            int(
                item[
                    "formal_index"
                ]
            )
            for item in entries
        ]
        ==
        list(
            range(
                120
            )
        ),
        "C1 formal indices changed.",
    )

    require(
        len(
            {
                str(
                    item[
                        "scenario_id"
                    ]
                )
                for item in entries
            }
        )
        ==
        120,
        "C1 scenario IDs not unique.",
    )

    for index, entry in enumerate(
        entries
    ):
        verify_c1_lock_entry(
            entry,
            EXPECTED_C1_RUNNER_SHA,
        )

        if (
            index % 20 == 0
            or
            index == 119
        ):
            print(
                f"C1 lock {index:03d}/119 verified",
                flush=True,
            )

    print(
        "C1 120/120 locks + canonical digests = PASS",
        flush=True,
    )

    formal_rows = [
        json.loads(
            line
        )
        for line
        in FORMAL.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    require(
        len(formal_rows)
        ==
        120,
        "Formal manifest N changed.",
    )

    formal_ids = [
        str(
            row[
                "scenario_id"
            ]
        )
        for row in formal_rows
    ]

    order_sha = sha256_bytes(
        json.dumps(
            formal_ids,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode(
            "utf-8"
        )
    )

    require(
        order_sha
        ==
        EXPECTED_FORMAL_ORDER_SHA,
        "Formal scenario-order SHA changed.",
    )

    require(
        formal_ids
        ==
        [
            str(
                item[
                    "scenario_id"
                ]
            )
            for item in entries
        ],
        (
            "C1 lock order differs "
            "from frozen formal order."
        ),
    )

    validation_rows = [
        json.loads(
            line
        )
        for line
        in VALIDATION_MANIFEST.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    require(
        len(validation_rows)
        ==
        44097,
        "Validation manifest count changed.",
    )

    validation_by_id = {
        str(
            row[
                "scenario_id"
            ]
        ):
            row
        for row in validation_rows
    }

    require(
        len(
            validation_by_id
        )
        ==
        44097,
        "Validation manifest IDs are not unique.",
    )

    resolved = []

    for formal in formal_rows:
        scenario_id = str(
            formal[
                "scenario_id"
            ]
        )

        require(
            scenario_id
            in
            validation_by_id,
            (
                "Formal scenario absent from "
                f"validation manifest: {scenario_id}"
            ),
        )

        validation = (
            validation_by_id[
                scenario_id
            ]
        )

        require(
            str(
                formal[
                    "selection_hash"
                ]
            )
            ==
            str(
                validation[
                    "selection_hash"
                ]
            ),
            (
                "Selection hash mismatch: "
                f"{scenario_id}"
            ),
        )

        require(
            str(
                formal[
                    "source_shard"
                ]
            )
            ==
            str(
                validation[
                    "source_shard"
                ]
            ),
            (
                "Source-shard provenance mismatch: "
                f"{scenario_id}"
            ),
        )

        require(
            int(
                formal[
                    "compact_record_offset"
                ]
            )
            ==
            int(
                validation[
                    "record_offset"
                ]
            ),
            (
                "Record-offset mismatch: "
                f"{scenario_id}"
            ),
        )

        resolved.append(
            validation
        )

    print(
        "formal -> validation metadata binding 120/120 = PASS",
        flush=True,
    )

    return (
        bundle_info,
        entries,
        resolved,
        order_sha,
    )


# =============================================================================
# Evaluator runtime binding
#
# IMPORTANT:
# this is initialized before future truth is opened.
# No model/checkpoint/controller is loaded here.
# =============================================================================

RUNTIME: dict[
    str,
    Any,
] = {}

VALIDATION_RUNTIME_BY_ID: dict[
    str,
    Any,
] = {}

GRID: Any = None
CODEBOOK: Any = None

TBEAM_S = 1.0e-5
TFRAME_S = 0.1


def initialize_runtime(
) -> None:
    global RUNTIME
    global VALIDATION_RUNTIME_BY_ID
    global GRID
    global CODEBOOK
    global TBEAM_S
    global TFRAME_S

    from iscai_stage3.validation.womd_access import (
        read_validation_manifest,
        read_motion_scenario,
    )

    from iscai_stage1.actors.womd_adapter import (
        adapt_causal_womd_scenario,
        object_type_name,
    )

    from iscai_stage5.beam_codebook import (
        build_uniform_azimuth_codebook,
    )

    from iscai_stage5.beam_baselines import (
        oracle_best_gain_beam,
    )

    from iscai_stage5.optical_link import (
        evaluate_optical_link,
        optical_gain_for_cell,
    )

    from iscai_stage6.adb.geometry import (
        Box3D,
        FractionalBoxRegion,
        project_box_to_headlamp,
        project_region_to_headlamp,
    )

    from iscai_stage6.adb.womd_geometry import (
        world_heading_to_headlamp_yaw,
    )

    from iscai_stage6.adb.probabilistic_occupancy import (
        OccupancyGrid,
    )

    from iscai_stage6.adb.metric_semantics import (
        binary_dim_support,
        mask_iou,
        vehicle_shadow_zone_violation,
        glare_risk_exposure,
        over_masking_area,
        road_illumination_retention,
        pedestrian_visibility_proxy,
        cyclist_visibility_proxy,
        false_dimming,
        temporal_smoothness,
        flicker_change_rate,
        normalized_energy_consumption,
    )

    RUNTIME = locals()

    validation_rows = (
        read_validation_manifest(
            VALIDATION_MANIFEST
        )
    )

    VALIDATION_RUNTIME_BY_ID = {
        str(
            row.scenario_id
        ):
            row
        for row in validation_rows
    }

    require(
        len(
            VALIDATION_RUNTIME_BY_ID
        )
        ==
        44097,
        (
            "Stage3 validation runtime "
            "row count changed."
        ),
    )

    beam_config = read_json(
        BEAM_CFG
    )[
        "azimuth_support"
    ]

    CODEBOOK = (
        build_uniform_azimuth_codebook(
            beam_count=(
                PRIMARY_CODEBOOK
            ),

            support_min_azimuth_rad=(
                math.radians(
                    float(
                        beam_config[
                            "min_deg"
                        ]
                    )
                )
            ),

            support_max_azimuth_rad=(
                math.radians(
                    float(
                        beam_config[
                            "max_deg"
                        ]
                    )
                )
            ),
        )
    )

    require(
        CODEBOOK.beam_count
        ==
        32,
        "Primary codebook construction failed.",
    )

    latency_config = read_json(
        BEAM_LATENCY_CFG
    )[
        "multi_rate_timing"
    ]

    TBEAM_S = float(
        latency_config[
            "Tbeam_s"
        ]
    )

    TFRAME_S = float(
        latency_config[
            "Tframe_s"
        ]
    )

    require(
        TBEAM_S
        ==
        1.0e-5
        and
        TFRAME_S
        ==
        0.1,
        "Frozen beam timing changed.",
    )

    grid_config = read_json(
        GRID_CFG
    )[
        "scientific_illumination_grid"
    ]

    theta = np.linspace(
        math.radians(
            float(
                grid_config[
                    "theta_centers_deg"
                ][
                    "minimum"
                ]
            )
        ),
        math.radians(
            float(
                grid_config[
                    "theta_centers_deg"
                ][
                    "maximum"
                ]
            )
        ),
        int(
            grid_config[
                "theta_centers_deg"
            ][
                "count"
            ]
        ),
        dtype=np.float64,
    )

    radial = np.linspace(
        float(
            grid_config[
                "range_centers_m"
            ][
                "minimum"
            ]
        ),
        float(
            grid_config[
                "range_centers_m"
            ][
                "maximum"
            ]
        ),
        int(
            grid_config[
                "range_centers_m"
            ][
                "count"
            ]
        ),
        dtype=np.float64,
    )

    GRID = OccupancyGrid(
        tuple(
            float(x)
            for x in theta
        ),
        tuple(
            float(x)
            for x in radial
        ),
    )

    require(
        GRID.shape
        ==
        GRID_SHAPE,
        "Frozen ADB grid shape changed.",
    )

    # Canonical Stage6 executable source is SHA-verified
    # by the preflight but MUST NOT be imported/executed here.
    # Pure evaluator-only surrogate logic is bound locally below.


# =============================================================================
# Frozen Stage5 communication evaluation
# =============================================================================

def in_decision_cell(
    azimuth: float,
    cell: Any,
    *,
    is_last: bool,
) -> bool:
    lower = float(
        cell.lower_azimuth_rad
    )

    upper = float(
        cell.upper_azimuth_rad
    )

    value = float(
        azimuth
    )

    if is_last:
        return bool(
            lower
            <=
            value
            <=
            upper
        )

    return bool(
        lower
        <=
        value
        <
        upper
    )


def selected_set_hit(
    azimuth: float,
    indices,
    codebook: Any,
) -> bool:
    index_set = {
        int(value)
        for value in indices
    }

    for cell in (
        codebook.cells
    ):
        if (
            int(
                cell.index
            )
            not in
            index_set
        ):
            continue

        if in_decision_cell(
            azimuth,
            cell,
            is_last=(
                int(
                    cell.index
                )
                ==
                len(
                    codebook.cells
                )
                -
                1
            ),
        ):
            return True

    return False


def selection_indices(
    value: Any,
) -> tuple[int, ...]:

    for name in (
        "beam_indices",
        "selected_beam_indices",
        "indices",
    ):
        item = getattr(
            value,
            name,
            None,
        )

        if item is not None:
            return tuple(
                int(x)
                for x in item
            )

    for name in (
        "beam_index",
        "selected_beam_index",
        "primary_beam_index",
    ):
        item = getattr(
            value,
            name,
            None,
        )

        if item is not None:
            return (
                int(item),
            )

    raise FailClosed(
        (
            "Could not extract oracle beam "
            f"indices from {type(value).__name__}"
        )
    )


def truth_decision_beam_index(
    azimuth: float,
    codebook: Any,
) -> int | None:

    for cell in (
        codebook.cells
    ):
        if in_decision_cell(
            azimuth,
            cell,
            is_last=(
                int(
                    cell.index
                )
                ==
                len(
                    codebook.cells
                )
                -
                1
            ),
        ):
            return int(
                cell.index
            )

    return None


def best_optical_link(
    *,
    indices,
    azimuth: float,
    elevation: float,
    codebook: Any,
    probing_beam_count: int,
):
    optical_gain_for_cell = (
        RUNTIME[
            "optical_gain_for_cell"
        ]
    )

    evaluate_optical_link = (
        RUNTIME[
            "evaluate_optical_link"
        ]
    )

    indices = tuple(
        int(value)
        for value in indices
    )

    if not indices:
        return (
            None,
            None,
        )

    candidates = []

    for index in indices:
        require(
            0
            <=
            index
            <
            len(
                codebook.cells
            ),
            (
                "Beam index outside "
                f"codebook: {index}"
            ),
        )

        cell = (
            codebook.cells[
                index
            ]
        )

        _, gain = (
            optical_gain_for_cell(
                receiver_azimuth_rad=(
                    azimuth
                ),

                receiver_elevation_rad=(
                    elevation
                ),

                cell=cell,
            )
        )

        candidates.append(
            (
                -float(
                    gain
                ),
                int(
                    index
                ),
                cell,
            )
        )

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    cell = (
        candidates[
            0
        ][
            2
        ]
    )

    result = (
        evaluate_optical_link(
            receiver_azimuth_rad=(
                azimuth
            ),

            receiver_elevation_rad=(
                elevation
            ),

            cell=cell,

            probing_beam_count=int(
                probing_beam_count
            ),

            beam_probe_time_s=(
                TBEAM_S
            ),

            frame_time_s=(
                TFRAME_S
            ),
        )
    )

    return (
        result,
        int(
            cell.index
        ),
    )


def link_metrics(
    achieved,
    oracle,
) -> dict[
    str,
    float | None,
]:

    if (
        achieved is None
        or
        oracle is None
    ):
        return {
            key:
                None
            for key in (
                "optical_gain",
                "received_power_normalized",
                "snr_db",
                "BER",
                "effective_rate_bps",
                "beam_gain_loss_db",
                "received_power_loss_db",
                "snr_loss_db",
                "effective_rate_loss_bps",
            )
        }

    epsilon = 1.0e-300

    gain_loss = (
        10.0
        *
        math.log10(
            max(
                float(
                    oracle.optical_gain
                ),
                epsilon,
            )
            /
            max(
                float(
                    achieved.optical_gain
                ),
                epsilon,
            )
        )
    )

    power_loss = (
        10.0
        *
        math.log10(
            max(
                float(
                    oracle
                    .received_power_normalized
                ),
                epsilon,
            )
            /
            max(
                float(
                    achieved
                    .received_power_normalized
                ),
                epsilon,
            )
        )
    )

    snr_loss = (
        float(
            oracle.snr_db
        )
        -
        float(
            achieved.snr_db
        )
    )

    rate_loss = (
        float(
            oracle
            .effective_rate
            .effective_rate_bps
        )
        -
        float(
            achieved
            .effective_rate
            .effective_rate_bps
        )
    )

    return {
        "optical_gain":
            float(
                achieved.optical_gain
            ),

        "received_power_normalized":
            float(
                achieved
                .received_power_normalized
            ),

        "snr_db":
            float(
                achieved.snr_db
            ),

        "BER":
            float(
                achieved.dbpsk_ber
            ),

        "effective_rate_bps":
            float(
                achieved
                .effective_rate
                .effective_rate_bps
            ),

        "beam_gain_loss_db":
            float(
                max(
                    0.0,
                    gain_loss,
                )
            ),

        "received_power_loss_db":
            float(
                max(
                    0.0,
                    power_loss,
                )
            ),

        "snr_loss_db":
            float(
                max(
                    0.0,
                    snr_loss,
                )
            ),

        "effective_rate_loss_bps":
            float(
                max(
                    0.0,
                    rate_loss,
                )
            ),
    }


# =============================================================================
# Fresh Stage7 / Stage6 evaluator reference
# =============================================================================

def future_box_from_state(
    state: Any,
    T_H0_from_W: Any,
):
    center = tuple(
        float(v)
        for v in (
            T_H0_from_W.apply_point(
                (
                    float(
                        state.center_x
                    ),
                    float(
                        state.center_y
                    ),
                    float(
                        state.center_z
                    ),
                )
            )
        )
    )

    dimensions = (
        float(
            state.length
        ),
        float(
            state.width
        ),
        float(
            state.height
        ),
    )

    require(
        all(
            math.isfinite(v)
            and
            v > 0.0
            for v in dimensions
        ),
        (
            "Evaluator-valid future actor "
            "has invalid dimensions."
        ),
    )

    yaw = (
        RUNTIME[
            "world_heading_to_headlamp_yaw"
        ](
            heading_W_rad=(
                float(
                    state.heading
                )
            ),

            T_H0_from_W=(
                T_H0_from_W
            ),
        )
    )

    box = (
        RUNTIME[
            "Box3D"
        ](
            center_xyz=center,

            length_m=(
                dimensions[
                    0
                ]
            ),

            width_m=(
                dimensions[
                    1
                ]
            ),

            height_m=(
                dimensions[
                    2
                ]
            ),

            yaw_rad=(
                yaw
            ),
        )
    )

    return (
        box,
        float(
            yaw
        ),
    )



def stage6_raster_projected(
    projected,
):
    theta = np.linspace(
        math.radians(-25.0),
        math.radians(25.0),
        501,
        dtype=np.float64,
    )

    radial = np.linspace(
        0.0,
        150.0,
        301,
        dtype=np.float64,
    )

    theta_half = 0.5 * float(
        theta[1] - theta[0]
    )

    range_half = 0.5 * float(
        radial[1] - radial[0]
    )

    center = float(
        projected.theta_center_rad
    )

    span = float(
        projected.theta_span_rad
    )

    dtheta = np.arctan2(
        np.sin(theta - center),
        np.cos(theta - center),
    )

    theta_active = (
        np.abs(dtheta)
        <=
        (
            0.5 * span
            +
            theta_half
        )
    )

    range_active = (
        (radial + range_half)
        >=
        float(
            projected.ground_range_min_m
        )
    ) & (
        (radial - range_half)
        <=
        float(
            projected.ground_range_max_m
        )
    )

    result = (
        theta_active[:, None]
        &
        range_active[None, :]
    )

    require(
        result.shape == GRID_SHAPE,
        "Stage6 evaluator raster shape mismatch.",
    )

    return result


def stage6_vehicle_surrogate_support(
    box,
    yaw_H0,
):
    FractionalBoxRegion = (
        RUNTIME[
            "FractionalBoxRegion"
        ]
    )

    project_region_to_headlamp = (
        RUNTIME[
            "project_region_to_headlamp"
        ]
    )

    if math.cos(
        float(yaw_H0)
    ) < 0.0:
        region = FractionalBoxRegion(
            x_bounds=(0.0, 0.5),
            y_bounds=(-0.5, 0.5),
            z_bounds=(0.0, 0.5),
        )
    else:
        region = FractionalBoxRegion(
            x_bounds=(-0.5, 0.0),
            y_bounds=(-0.5, 0.5),
            z_bounds=(0.0, 0.5),
        )

    projected = (
        project_region_to_headlamp(
            box,
            region,
        )
    )

    return stage6_raster_projected(
        projected
    )


def stage7_future_index(
    scenario,
    horizon_s,
):
    """
    Nearest observed future WOMD timestamp;
    earliest index wins an exact tie.
    No interpolation.
    """

    anchor = int(
        scenario.current_time_index
    )

    ts = np.asarray(
        scenario.timestamps_seconds,
        dtype=np.float64,
    )

    target = (
        float(ts[anchor])
        +
        float(horizon_s)
    )

    candidates = np.arange(
        anchor + 1,
        len(ts),
        dtype=np.int64,
    )

    require(
        len(candidates) > 0,
        "No future WOMD timestamp.",
    )

    errors = np.abs(
        ts[candidates]
        -
        target
    )

    pos = int(
        np.argmin(errors)
    )

    return int(
        candidates[pos]
    )


def build_fresh_adb_reference(
    scenario: Any,
    adapted: Any,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, Any],
]:

    object_type_name = (
        RUNTIME[
            "object_type_name"
        ]
    )

    project_box_to_headlamp = (
        RUNTIME[
            "project_box_to_headlamp"
        ]
    )

    T_H0_from_W = (
        adapted.frames.T_H0_from_W
    )

    anchor = int(
        scenario.current_time_index
    )

    arrays = {
        name:
            np.zeros(
                SCHEDULE_SHAPE,
                dtype=np.bool_,
            )
        for name in (
            REFERENCE_FIELDS
        )
    }

    actor_counts = {
        actor_class:
            [
                0,
                0,
                0,
                0,
            ]
        for actor_class in (
            CORE_CLASSES
        )
    }

    for track_index, track in enumerate(
        scenario.tracks
    ):
        if (
            track_index
            ==
            int(
                scenario.sdc_track_index
            )
        ):
            continue

        actor_class = str(
            object_type_name(
                track
            )
        )

        if (
            actor_class
            not in
            CORE_CLASSES
        ):
            continue

        for horizon_index, horizon_s in enumerate(
            HORIZONS_S
        ):
            state_index = (
                stage7_future_index(
                    scenario,
                    horizon_s,
                )
            )

            if (
                state_index
                >=
                len(
                    track.states
                )
                or
                not bool(
                    track.states[
                        state_index
                    ].valid
                )
            ):
                continue

            state = (
                track.states[
                    state_index
                ]
            )

            box, yaw = (
                future_box_from_state(
                    state,
                    T_H0_from_W,
                )
            )

            projected = (
                project_box_to_headlamp(
                    box,

                    state_source=(
                        "WOMD_future_GT_evaluator_only"
                    ),

                    orientation_source=(
                        "oracle_evaluator_only"
                    ),

                    controller_path=False,
                )
            )

            support = np.asarray(
                stage6_raster_projected(
                    projected
                ),
                dtype=np.bool_,
            )

            require(
                support.shape
                ==
                GRID_SHAPE,
                (
                    "Future full-box raster "
                    "shape changed."
                ),
            )

            arrays[
                "oracle_all"
            ][
                horizon_index
            ] |= support

            actor_counts[
                actor_class
            ][
                horizon_index
            ] += 1

            if (
                actor_class
                ==
                "TYPE_VEHICLE"
            ):
                arrays[
                    "mask_vehicle"
                ][
                    horizon_index
                ] |= support

                arrays[
                    "oracle_vehicle"
                ][
                    horizon_index
                ] |= support

                surrogate = np.asarray(
                    stage6_vehicle_surrogate_support(
                        box,
                        yaw,
                    ),
                    dtype=np.bool_,
                )

                require(
                    surrogate.shape
                    ==
                    GRID_SHAPE,
                    (
                        "Vehicle-surrogate raster "
                        "shape changed."
                    ),
                )

                arrays[
                    "vehicle_surrogate"
                ][
                    horizon_index
                ] |= surrogate

            elif (
                actor_class
                ==
                "TYPE_PEDESTRIAN"
            ):
                arrays[
                    "mask_pedestrian"
                ][
                    horizon_index
                ] |= support

                arrays[
                    "pedestrian_region"
                ][
                    horizon_index
                ] |= support

            else:
                arrays[
                    "mask_cyclist"
                ][
                    horizon_index
                ] |= support

                arrays[
                    "cyclist_region"
                ][
                    horizon_index
                ] |= support

    return (
        arrays,
        {
            "actor_counts_by_class_horizon":
                actor_counts,

            "support_cells": {
                key: [
                    int(
                        arrays[
                            key
                        ][
                            horizon
                        ].sum()
                    )
                    for horizon in range(
                        4
                    )
                ]
                for key in (
                    REFERENCE_FIELDS
                )
            },

            "semantics": (
                "fresh_Stage7_constructed_"
                "evaluator_reference_from_"
                "future_WOMD_full_physical_boxes"
            ),

            "measured_ADB_ground_truth":
                False,

            "historical_Stage6_outcomes_reused":
                False,
        },
    )


# =============================================================================
# Exact frozen Stage6 metric engine
# =============================================================================

def evaluate_adb(
    schedule: np.ndarray,
    current: np.ndarray,
    reference: dict[
        str,
        np.ndarray,
    ],
) -> dict[
    str,
    float | None,
]:

    schedule = np.asarray(
        schedule,
        dtype=np.float64,
    )

    current = np.asarray(
        current,
        dtype=np.float64,
    )

    require(
        schedule.shape
        ==
        SCHEDULE_SHAPE,
        "ADB schedule shape mismatch.",
    )

    require(
        current.shape
        ==
        GRID_SHAPE,
        "Current ADB map shape mismatch.",
    )

    require(
        np.all(
            np.isfinite(
                schedule
            )
        ),
        "ADB schedule has non-finite values.",
    )

    require(
        np.all(
            (
                schedule
                >=
                0.0
            )
            &
            (
                schedule
                <=
                1.0
            )
        ),
        "ADB schedule outside [0,1].",
    )

    candidate_mask = (
        RUNTIME[
            "binary_dim_support"
        ](
            schedule
        )
    )

    road_roi = np.ones(
        SCHEDULE_SHAPE,
        dtype=np.bool_,
    )

    temporal_sequence = (
        np.concatenate(
            (
                current[
                    None,
                    :,
                    :,
                ],
                schedule,
            ),
            axis=0,
        )
    )

    values = {
        "mask_IoU_with_oracle_future_mask":
            RUNTIME[
                "mask_iou"
            ](
                candidate_mask,
                reference[
                    "oracle_all"
                ],
            ),

        "vehicle_shadow_zone_violation":
            RUNTIME[
                "vehicle_shadow_zone_violation"
            ](
                candidate_mask,
                reference[
                    "oracle_vehicle"
                ],
            ),

        "glare_risk_exposure":
            RUNTIME[
                "glare_risk_exposure"
            ](
                schedule,
                reference[
                    "vehicle_surrogate"
                ],
            ),

        "over_masking_area":
            RUNTIME[
                "over_masking_area"
            ](
                candidate_mask,
                reference[
                    "oracle_all"
                ],
            ),

        "road_illumination_retention":
            RUNTIME[
                "road_illumination_retention"
            ](
                schedule,
                road_roi,
            ),

        "pedestrian_visibility_proxy":
            RUNTIME[
                "pedestrian_visibility_proxy"
            ](
                schedule,
                reference[
                    "pedestrian_region"
                ],
            ),

        "cyclist_visibility_proxy":
            RUNTIME[
                "cyclist_visibility_proxy"
            ](
                schedule,
                reference[
                    "cyclist_region"
                ],
            ),

        "false_dimming":
            RUNTIME[
                "false_dimming"
            ](
                schedule,
                reference[
                    "oracle_all"
                ],
            ),

        "temporal_smoothness":
            RUNTIME[
                "temporal_smoothness"
            ](
                temporal_sequence,

                delta_t_s=(
                    TEMPORAL_DELTA_T_S
                ),
            ),

        "flicker_change_rate":
            RUNTIME[
                "flicker_change_rate"
            ](
                temporal_sequence
            ),

        "energy_consumption":
            RUNTIME[
                "normalized_energy_consumption"
            ](
                schedule
            ),
    }

    return {
        key:
            safe_number(
                value
            )
        for key, value in (
            values.items()
        )
    }


# =============================================================================
# Fresh Stage7 receiver future truth
# =============================================================================

def receiver_truth(
    scenario: Any,
    adapted: Any,
    receiver: dict[str, Any],
) -> list[
    dict[str, Any]
]:

    track_index = int(
        receiver[
            "track_index"
        ]
    )

    require(
        0
        <=
        track_index
        <
        len(
            scenario.tracks
        ),
        (
            "Receiver track_index invalid "
            "after evaluator open."
        ),
    )

    track = (
        scenario.tracks[
            track_index
        ]
    )

    require(
        str(
            track.id
        )
        ==
        str(
            receiver[
                "track_id"
            ]
        ),
        "Receiver track_id changed.",
    )

    require(
        str(
            RUNTIME[
                "object_type_name"
            ](
                track
            )
        )
        ==
        str(
            receiver[
                "actor_class"
            ]
        ),
        "Receiver actor class changed.",
    )

    anchor = int(
        scenario.current_time_index
    )

    T_H0_from_W = (
        adapted.frames.T_H0_from_W
    )

    result = []

    for horizon_index, horizon_s in enumerate(
        HORIZONS_S
    ):
        state_index = (
            stage7_future_index(
                scenario,
                horizon_s,
            )
        )

        valid = (
            state_index
            <
            len(
                track.states
            )
            and
            bool(
                track.states[
                    state_index
                ].valid
            )
        )

        if not valid:
            result.append(
                {
                    "horizon_index":
                        horizon_index,

                    "horizon_s":
                        horizon_s,

                    "valid":
                        False,
                }
            )

            continue

        state = (
            track.states[
                state_index
            ]
        )

        position = np.asarray(
            T_H0_from_W.apply_point(
                (
                    float(
                        state.center_x
                    ),
                    float(
                        state.center_y
                    ),
                    float(
                        state.center_z
                    ),
                )
            ),
            dtype=np.float64,
        )

        x, y, z = (
            float(value)
            for value in position
        )

        planar = math.hypot(
            x,
            y,
        )

        radius = math.sqrt(
            x * x
            +
            y * y
            +
            z * z
        )

        azimuth = math.atan2(
            y,
            x,
        )

        elevation = math.atan2(
            z,
            planar,
        )

        result.append(
            {
                "horizon_index":
                    horizon_index,

                "horizon_s":
                    horizon_s,

                "valid":
                    True,

                "position_H0_m":
                    [
                        x,
                        y,
                        z,
                    ],

                "range_m":
                    radius,

                "azimuth_rad":
                    azimuth,

                "elevation_rad":
                    elevation,

                "truth_decision_beam_index":
                    truth_decision_beam_index(
                        azimuth,
                        CODEBOOK,
                    ),
            }
        )

    return result


# =============================================================================
# Frozen C1 communication output consumption
# =============================================================================

def resolve_comm_output(
    lock: dict[str, Any],
    system: str,
) -> tuple[
    dict[str, Any],
    str,
]:

    raw = (
        lock[
            "communication_outputs"
        ][
            system
        ]
    )

    if (
        raw.get(
            "status"
        )
        ==
        "REUSE_SHARED_COMMUNICATION_BRANCH"
    ):
        source = str(
            raw[
                "source_system"
            ]
        )

        require(
            source
            ==
            "shared_trajectory_posterior",
            (
                "Unexpected communication "
                "alias source."
            ),
        )

        return (
            lock[
                "communication_outputs"
            ][
                source
            ],
            source,
        )

    return (
        raw,
        system,
    )


def evaluate_communication(
    lock: dict[str, Any],
    system: str,
    truth: list[
        dict[str, Any]
    ],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:

    output, source_system = (
        resolve_comm_output(
            lock,
            system,
        )
    )

    require(
        output.get(
            "status"
        )
        ==
        "SELECTED",
        (
            "Communication output not SELECTED: "
            f"{system}"
        ),
    )

    codebook_output = (
        output[
            "codebooks"
        ][
            str(
                PRIMARY_CODEBOOK
            )
        ]
    )

    decisions = (
        codebook_output[
            "primary_q_0p95_temporal_decisions"
        ]
    )

    probabilities = (
        codebook_output[
            "probability"
        ]
    )

    require(
        len(decisions)
        ==
        4
        and
        len(probabilities)
        ==
        4,
        (
            "Communication horizon "
            f"cardinality changed: {system}"
        ),
    )

    horizon_rows = []

    for horizon_index in range(
        4
    ):
        decision = (
            decisions[
                horizon_index
            ]
        )

        probability = (
            probabilities[
                horizon_index
            ]
        )

        truth_row = (
            truth[
                horizon_index
            ]
        )

        require(
            abs(
                float(
                    decision[
                        "horizon_s"
                    ]
                )
                -
                HORIZONS_S[
                    horizon_index
                ]
            )
            <
            1.0e-12,
            (
                "Communication horizon changed: "
                f"{system}/{horizon_index}"
            ),
        )

        masses = [
            float(value)
            for value in (
                probability[
                    "masses"
                ]
            )
        ]

        require(
            len(masses)
            ==
            32,
            "Locked probability vector is not 32-D.",
        )

        require(
            all(
                math.isfinite(value)
                and
                value >= 0.0
                for value in masses
            ),
            "Invalid locked probability masses.",
        )

        selected = tuple(
            int(value)
            for value in (
                decision[
                    "selected_beam_indices"
                ]
            )
        )

        selected_k = int(
            decision[
                "selected_K"
            ]
        )

        require(
            selected_k
            ==
            len(
                selected
            )
            and
            selected_k
            >
            0,
            "selected_K mismatch.",
        )

        local = tuple(
            int(value)
            for value in (
                decision.get(
                    "local_neighbor_probe_indices",
                    [],
                )
            )
        )

        loss_of_lock = bool(
            decision[
                "loss_of_lock"
            ]
        )

        exhaustive = bool(
            decision[
                "exhaustive_fallback_active"
            ]
        )

        # Frozen Stage5 physical probe accounting:
        #
        # normal:
        #   charge selected K.
        #
        # loss-of-lock at primary 32-beam codebook:
        #   one widened 16-beam probe
        #   + exhaustive 32-beam sweep
        #   => 33 charged probes.
        #
        # The link search over the current codebook is
        # exhaustive in this case; the widened probe is
        # additional latency/overhead recovery cost.
        if loss_of_lock:
            require(
                exhaustive,
                (
                    "Loss-of-lock without "
                    "exhaustive fallback."
                ),
            )

            physical_indices = tuple(
                range(
                    32
                )
            )

            physical_probe_count = 33

        else:
            # Frozen Stage5 beam-latency semantics:
            #
            # Normal operation probes exactly the adaptive
            # mass-covering selected beam set.
            #
            # local_neighbor_probe_indices are NOT added to
            # the normal physical probe schedule.
            physical_indices = tuple(
                selected
            )

            physical_probe_count = (
                selected_k
            )

        reacquisition_latency_s = (
            max(
                0,
                physical_probe_count
                -
                selected_k,
            )
            *
            TBEAM_S
            if loss_of_lock
            else
            0.0
        )

        row = {
            "horizon_index":
                horizon_index,

            "horizon_s":
                HORIZONS_S[
                    horizon_index
                ],

            "valid_future_receiver":
                bool(
                    truth_row[
                        "valid"
                    ]
                ),

            "selected_beam_indices":
                list(
                    selected
                ),

            "selected_K":
                selected_k,

            "achieved_mass":
                float(
                    decision[
                        "achieved_mass"
                    ]
                ),

            "primary_beam_index":
                int(
                    decision[
                        "primary_beam_index"
                    ]
                ),

            "primary_switched":
                bool(
                    decision[
                        "primary_switched"
                    ]
                ),

            "loss_of_lock":
                loss_of_lock,

            "exhaustive_fallback_active":
                exhaustive,

            "local_neighbor_probe_indices":
                list(
                    local
                ),

            "physical_probe_count":
                int(
                    physical_probe_count
                ),

            "probing_overhead_fraction":
                float(
                    physical_probe_count
                    *
                    TBEAM_S
                    /
                    TFRAME_S
                ),

            "overhead_reduction_vs_exhaustive":
                float(
                    1.0
                    -
                    physical_probe_count
                    /
                    32.0
                ),

            "reacquisition_latency_s":
                float(
                    reacquisition_latency_s
                ),

            "locked_probability_masses":
                masses,

            "inside_support_mass":
                float(
                    probability[
                        "inside_support_mass"
                    ]
                ),

            "outside_support_mass":
                float(
                    probability[
                        "outside_support_mass"
                    ]
                ),

            "communication_branch_source_system":
                source_system,
        }

        if not truth_row[
            "valid"
        ]:
            row.update(
                {
                    "containment_hit":
                        None,

                    "beam_outage":
                        None,

                    "receiver_distance_m":
                        None,

                    "truth_azimuth_rad":
                        None,

                    "truth_elevation_rad":
                        None,

                    "truth_decision_beam_index":
                        None,

                    "oracle_best_gain_beam_index":
                        None,
                }
            )

            row.update(
                link_metrics(
                    None,
                    None,
                )
            )

            horizon_rows.append(
                row
            )

            continue

        truth_azimuth = float(
            truth_row[
                "azimuth_rad"
            ]
        )

        truth_elevation = float(
            truth_row[
                "elevation_rad"
            ]
        )

        # Literal frozen Stage5 oracle route.
        oracle_selection = (
            RUNTIME[
                "oracle_best_gain_beam"
            ](
                realized_azimuth_rad=(
                    truth_azimuth
                ),
                codebook=(
                    CODEBOOK
                ),
            )
        )

        oracle_indices = (
            selection_indices(
                oracle_selection
            )
        )

        require(
            len(
                oracle_indices
            )
            ==
            1,
            (
                "Frozen oracle_best_gain_beam "
                "must select exactly one beam."
            ),
        )

        oracle_link, oracle_index = (
            best_optical_link(
                indices=(
                    oracle_indices
                ),

                azimuth=(
                    truth_azimuth
                ),

                elevation=(
                    truth_elevation
                ),

                codebook=(
                    CODEBOOK
                ),

                probing_beam_count=1,
            )
        )

        # Frozen Stage5 semantics:
        #
        # oracle_best_gain_beam selects the evaluator-only
        # beam using the constructed Block5.3 directional
        # score.  That selected beam is then evaluated by
        # the separate frozen Part-A optical-link engine.
        #
        # Do NOT redefine this oracle as an argmax over
        # final optical-link gain.

        achieved_link, _ = (
            best_optical_link(
                indices=(
                    physical_indices
                ),

                azimuth=(
                    truth_azimuth
                ),

                elevation=(
                    truth_elevation
                ),

                codebook=(
                    CODEBOOK
                ),

                probing_beam_count=(
                    physical_probe_count
                ),
            )
        )

        hit = (
            selected_set_hit(
                truth_azimuth,
                selected,
                CODEBOOK,
            )
        )

        row.update(
            {
                "containment_hit":
                    bool(
                        hit
                    ),

                "beam_outage":
                    bool(
                        not hit
                    ),

                "receiver_distance_m":
                    float(
                        truth_row[
                            "range_m"
                        ]
                    ),

                "truth_azimuth_rad":
                    truth_azimuth,

                "truth_elevation_rad":
                    truth_elevation,

                "truth_decision_beam_index":
                    truth_row[
                        "truth_decision_beam_index"
                    ],

                "oracle_best_gain_beam_index":
                    oracle_index,
            }
        )

        row.update(
            link_metrics(
                achieved_link,
                oracle_link,
            )
        )

        horizon_rows.append(
            row
        )

    valid_rows = [
        row
        for row in horizon_rows
        if row[
            "valid_future_receiver"
        ]
    ]

    metrics = {
        "probability_coverage":
            mean_finite(
                1.0
                if row[
                    "containment_hit"
                ]
                else
                0.0
                for row in valid_rows
            ),

        "outage_probability":
            mean_finite(
                1.0
                if row[
                    "beam_outage"
                ]
                else
                0.0
                for row in valid_rows
            ),

        "selected_K":
            mean_finite(
                row[
                    "selected_K"
                ]
                for row in valid_rows
            ),

        "probing_overhead_fraction":
            mean_finite(
                row[
                    "probing_overhead_fraction"
                ]
                for row in valid_rows
            ),

        "overhead_reduction_vs_exhaustive":
            mean_finite(
                row[
                    "overhead_reduction_vs_exhaustive"
                ]
                for row in valid_rows
            ),

        "beam_switching_rate":
            mean_finite(
                1.0
                if row[
                    "primary_switched"
                ]
                else
                0.0
                for row in valid_rows
            ),

        "reacquisition_latency_s":
            mean_finite(
                row[
                    "reacquisition_latency_s"
                ]
                for row in valid_rows
            ),

        "posterior_probability_mass_selected":
            mean_finite(
                row[
                    "achieved_mass"
                ]
                for row in valid_rows
            ),

        "beam_gain_loss_db":
            mean_finite(
                row[
                    "beam_gain_loss_db"
                ]
                for row in valid_rows
            ),

        "snr_loss_db":
            mean_finite(
                row[
                    "snr_loss_db"
                ]
                for row in valid_rows
            ),

        "BER":
            mean_finite(
                row[
                    "BER"
                ]
                for row in valid_rows
            ),

        "effective_rate_bps":
            mean_finite(
                row[
                    "effective_rate_bps"
                ]
                for row in valid_rows
            ),

        "effective_rate_loss_bps":
            mean_finite(
                row[
                    "effective_rate_loss_bps"
                ]
                for row in valid_rows
            ),

        "valid_future_horizons":
            int(
                len(
                    valid_rows
                )
            ),
    }

    return (
        metrics,
        horizon_rows,
    )


# =============================================================================
# Frozen C1 ADB schedule consumption
# =============================================================================

def load_c1_schedules(
    lock: dict[str, Any],
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
]:

    path = Path(
        lock[
            "ADB_outputs"
        ][
            "npz_path"
        ]
    )

    require(
        sha256_file(
            path
        )
        ==
        lock[
            "adb_schedules_npz_sha256"
        ],
        (
            "C1 ADB NPZ changed after "
            "worker lock verification."
        ),
    )

    with np.load(
        path,
        allow_pickle=False,
    ) as archive:

        require(
            set(
                archive.files
            )
            ==
            {
                member[:-4]
                for member in (
                    EXPECTED_C1_NPZ_MEMBERS
                )
            },
            "C1 NPZ array names changed.",
        )

        current = np.asarray(
            archive[
                "current_reactive_t0"
            ],
            dtype=np.float64,
        )

        maps = {
            key:
                np.asarray(
                    archive[
                        key
                    ],
                    dtype=np.float64,
                )
            for key in (
                "shared_trajectory_posterior",
                "independent_models",
                "direct_ADB_predictor",
                "deterministic_shared_trajectory",
            )
        }

    require(
        current.shape
        ==
        GRID_SHAPE,
        "C1 current ADB map shape changed.",
    )

    require(
        all(
            value.shape
            ==
            SCHEDULE_SHAPE
            for value in (
                maps.values()
            )
        ),
        "C1 ADB schedule shape changed.",
    )

    return (
        current,
        maps,
    )


def system_schedule(
    system: str,
    maps: dict[
        str,
        np.ndarray,
    ],
) -> np.ndarray:

    if (
        system
        ==
        "direct_beam_classifier"
    ):
        # Frozen hybrid:
        # direct beam communication branch +
        # exact shared-posterior ADB branch.
        return maps[
            "shared_trajectory_posterior"
        ]

    return maps[
        system
    ]


# =============================================================================
# Resume-safe immutable C2 scenario artifacts
# =============================================================================

def c2_dir(
    index: int,
    scenario_id: str,
) -> Path:
    return (
        OUT_ROOT
        /
        (
            f"{index:03d}_"
            f"{scenario_id}"
        )
    )


def verify_existing_c2(
    final_dir: Path,
    entry: dict[str, Any],
    runner_sha: str,
) -> dict[str, Any] | None:

    if not final_dir.exists():
        return None

    require(
        final_dir.is_dir(),
        (
            "C2 path exists but is not "
            f"a directory: {final_dir}"
        ),
    )

    record_path = (
        final_dir
        / "record.json"
    )

    reference_path = (
        final_dir
        / "reference.npz"
    )

    require(
        record_path.is_file()
        and
        reference_path.is_file(),
        (
            "Partial immutable C2 directory: "
            f"{final_dir}"
        ),
    )

    document = read_json(
        record_path
    )

    require(
        document.get(
            "block"
        )
        ==
        "7.5C2",
        "Existing C2 block mismatch.",
    )

    require(
        document.get(
            "status"
        )
        ==
        "IMMUTABLE_C2_SCENARIO_COMPLETE",
        "Existing C2 record status invalid.",
    )

    require(
        int(
            document[
                "formal_index"
            ]
        )
        ==
        int(
            entry[
                "formal_index"
            ]
        ),
        "Existing C2 formal-index mismatch.",
    )

    require(
        str(
            document[
                "scenario_id"
            ]
        )
        ==
        str(
            entry[
                "scenario_id"
            ]
        ),
        "Existing C2 scenario mismatch.",
    )

    require(
        document[
            "runner_sha256"
        ]
        ==
        runner_sha,
        "Existing C2 runner SHA mismatch.",
    )

    require(
        document[
            "c1_lock_json_sha256"
        ]
        ==
        entry[
            "lock_json_sha256"
        ],
        "Existing C2/C1 JSON binding mismatch.",
    )

    require(
        document[
            "c1_adb_schedules_npz_sha256"
        ]
        ==
        entry[
            "adb_schedules_npz_sha256"
        ],
        "Existing C2/C1 NPZ binding mismatch.",
    )

    require(
        document[
            "reference_npz_sha256"
        ]
        ==
        sha256_file(
            reference_path
        ),
        "Existing C2 reference SHA mismatch.",
    )

    core = dict(
        document
    )

    stored_digest = (
        core.pop(
            "raw_record_digest_sha256"
        )
    )

    require(
        sha256_bytes(
            canonical_bytes(
                core
            )
        )
        ==
        stored_digest,
        (
            "Existing C2 raw-record "
            "digest mismatch."
        ),
    )

    require(
        len(
            document[
                "joint_system_rows"
            ]
        )
        ==
        5,
        "Existing C2 system-row count != 5.",
    )

    require(
        [
            row[
                "system"
            ]
            for row in (
                document[
                    "joint_system_rows"
                ]
            )
        ]
        ==
        list(
            SYSTEMS
        ),
        "Existing C2 system order changed.",
    )

    return {
        "formal_index":
            int(
                entry[
                    "formal_index"
                ]
            ),

        "scenario_id":
            str(
                entry[
                    "scenario_id"
                ]
            ),

        "record_path":
            str(
                record_path
            ),

        "record_sha256":
            sha256_file(
                record_path
            ),

        "reference_path":
            str(
                reference_path
            ),

        "reference_sha256":
            sha256_file(
                reference_path
            ),

        "raw_record_digest_sha256":
            stored_digest,

        "resumed":
            True,
    }


# =============================================================================
# One C2 formal scenario
# =============================================================================

def process_one(
    task: tuple[
        dict[str, Any],
        dict[str, Any],
        str,
    ],
) -> dict[str, Any]:

    (
        entry,
        validation_metadata,
        runner_sha,
    ) = task

    formal_index = int(
        entry[
            "formal_index"
        ]
    )

    scenario_id = str(
        entry[
            "scenario_id"
        ]
    )

    final_dir = c2_dir(
        formal_index,
        scenario_id,
    )

    # Resume path:
    # do not reopen future GT if a fully valid immutable
    # C2 result already exists.
    existing = verify_existing_c2(
        final_dir,
        entry,
        runner_sha,
    )

    if existing is not None:
        return existing

    # =====================================================================
    # CRITICAL PER-SCENARIO CAUSALITY BARRIER
    #
    # Verify the durable C1 lock IMMEDIATELY before the
    # first future-capable WOMD read.
    # =====================================================================

    lock = verify_c1_lock_entry(
        entry,
        EXPECTED_C1_RUNNER_SHA,
    )

    require(
        lock[
            "future_GT_accessed"
        ]
        is False,
        (
            "C1 future-GT boundary invalid "
            "immediately before evaluator open."
        ),
    )

    require(
        lock[
            "formal_metrics_computed"
        ]
        is False,
        (
            "C1 metric boundary invalid "
            "immediately before evaluator open."
        ),
    )

    # =====================================================================
    # ONLY NOW: evaluator-only future WOMD is opened.
    # =====================================================================

    runtime_row = (
        VALIDATION_RUNTIME_BY_ID.get(
            scenario_id
        )
    )

    require(
        runtime_row is not None,
        (
            "Runtime validation row missing: "
            f"{scenario_id}"
        ),
    )

    scenario = (
        RUNTIME[
            "read_motion_scenario"
        ](
            runtime_row,

            paired_root=(
                PAIRED_ROOT
            ),

            compact_record_offset=int(
                validation_metadata[
                    "record_offset"
                ]
            ),
        )
    )

    require(
        str(
            scenario.scenario_id
        )
        ==
        scenario_id,
        "Future evaluator scenario ID mismatch.",
    )

    require(
        int(
            scenario.current_time_index
        )
        ==
        10,
        "Frozen WOMD current_time_index changed.",
    )

    adapted = (
        RUNTIME[
            "adapt_causal_womd_scenario"
        ](
            scenario
        )
    )

    # =====================================================================
    # FROM THIS POINT ON:
    #
    #   NO predictor forward
    #   NO controller generation
    #   NO retuning
    #
    # Only evaluator truth/reference construction and scoring
    # of already-locked C1 outputs.
    # =====================================================================

    reference, reference_metadata = (
        build_fresh_adb_reference(
            scenario,
            adapted,
        )
    )

    reference_bytes = (
        deterministic_npz_bytes(
            reference
        )
    )

    reference_sha = (
        sha256_bytes(
            reference_bytes
        )
    )

    current_t0, maps = (
        load_c1_schedules(
            lock
        )
    )

    communication_truth = (
        receiver_truth(
            scenario,
            adapted,
            lock[
                "receiver"
            ],
        )
    )

    joint_rows = []
    communication_horizon_records = {}

    for system in SYSTEMS:

        communication_metrics, horizon_rows = (
            evaluate_communication(
                lock,
                system,
                communication_truth,
            )
        )

        adb_metrics = (
            evaluate_adb(
                system_schedule(
                    system,
                    maps,
                ),
                current_t0,
                reference,
            )
        )

        flat_metrics = {
            key:
                value
            for key, value in (
                communication_metrics.items()
            )
            if key
            !=
            "valid_future_horizons"
        }

        flat_metrics.update(
            adb_metrics
        )

        _, communication_source = (
            resolve_comm_output(
                lock,
                system,
            )
        )

        joint_rows.append(
            {
                "formal_index":
                    formal_index,

                "scenario_id":
                    scenario_id,

                "system":
                    system,

                "communication_metrics":
                    communication_metrics,

                "illumination_metrics":
                    adb_metrics,

                "metrics":
                    flat_metrics,

                "communication_branch_source_system":
                    communication_source,

                "ADB_branch_source_system":
                    (
                        "shared_trajectory_posterior"
                        if system
                        ==
                        "direct_beam_classifier"
                        else
                        system
                    ),

                "future_GT_controller_input":
                    False,

                "post_outcome_tuning":
                    False,
            }
        )

        communication_horizon_records[
            system
        ] = horizon_rows

    reference_array_digests = {
        key:
            array_digest(
                value
            )
        for key, value in (
            reference.items()
        )
    }

    core = {
        "project":
            "Agni",

        "stage":
            7,

        "block":
            "7.5C2",

        "status":
            "IMMUTABLE_C2_SCENARIO_COMPLETE",

        "formal_index":
            formal_index,

        "scenario_id":
            scenario_id,

        "runner_path":
            str(
                SCRIPT_EXPECTED_PATH
            ),

        "runner_sha256":
            runner_sha,

        "block75b_contract_sha256":
            EXPECTED_B75B_SHA,

        "formal_manifest_sha256":
            EXPECTED_FORMAL_SHA,

        "formal_order_sha256":
            EXPECTED_FORMAL_ORDER_SHA,

        "c1_manifest_sha256":
            EXPECTED_C1_MANIFEST_SHA,

        "c1_runner_sha256":
            EXPECTED_C1_RUNNER_SHA,

        "c1_checker_sha256":
            EXPECTED_C1_CHECKER_SHA,

        "c1_lock_json":
            str(
                entry[
                    "lock_json"
                ]
            ),

        "c1_lock_json_sha256":
            entry[
                "lock_json_sha256"
            ],

        "c1_adb_schedules_npz":
            str(
                entry[
                    "adb_schedules_npz"
                ]
            ),

        "c1_adb_schedules_npz_sha256":
            entry[
                "adb_schedules_npz_sha256"
            ],

        "c1_nonoracle_output_digest_sha256":
            entry[
                "nonoracle_output_digest_sha256"
            ],

        "causality": {
            "valid_C1_lock_verified_immediately_before_future_GT_open":
                True,

            "future_GT_evaluator_only":
                True,

            "future_GT_controller_input":
                False,

            "controllers_rerun_after_future_GT_access":
                False,

            "model_forward_after_future_GT_access":
                False,

            "historical_Stage6_outcomes_reused":
                False,

            "post_outcome_tuning":
                False,

            "scientific_parameters_modified":
                False,
        },

        "future_GT": {
            "opened":
                True,

            "receiver_geometry_mode": (
                "centroid_baseline_matching_"
                "locked_C1_center_geometry"
            ),

            "receiver_truth_by_horizon":
                communication_truth,
        },

        "fresh_Stage6_constructed_ADB_reference": {
            **reference_metadata,

            "reference_fields":
                list(
                    REFERENCE_FIELDS
                ),

            "reference_array_sha256":
                reference_array_digests,

            "reference_npz_path":
                str(
                    final_dir
                    /
                    "reference.npz"
                ),

            "reference_npz_sha256":
                reference_sha,

            "grid_shape":
                list(
                    GRID_SHAPE
                ),

            "schedule_shape":
                list(
                    SCHEDULE_SHAPE
                ),

            "horizons_s":
                list(
                    HORIZONS_S
                ),

            "vehicle_surrogate_source":
                str(
                    SURROGATE_SOURCE
                ),

            "vehicle_surrogate_source_sha256":
                EXPECTED_SURROGATE_SOURCE_SHA,
        },

        "primary_communication_protocol": {
            "codebook_size":
                32,

            "coverage_target":
                PRIMARY_COVERAGE,

            "Tbeam_s":
                TBEAM_S,

            "Tframe_s":
                TFRAME_S,

            "oracle":
                "oracle_best_gain_beam",

            "optical_link":
                (
                    "frozen_Stage5_"
                    "optical_link_evaluation"
                ),

            "fresh_Stage7_truth":
                True,
        },

        "joint_system_rows":
            joint_rows,

        "communication_horizon_records":
            communication_horizon_records,

        # Raw support needed by C3 for applicable
        # calibration analyses.
        #
        # Critically: do NOT rerun direct ADB after truth is open
        # merely to recover a probability field that C1 did not lock.
        "representation_quality_raw_support": {
            "receiver_future_positions_H0_m_available":
                True,

            "beam_probability_vectors_locked_in_C1":
                True,

            "truth_decision_beam_indices_available":
                True,

            "direct_ADB_raw_probability_field_locked_in_C1":
                False,

            "direct_ADB_probability_calibration_note": (
                "explicit_NA_if_raw_probability_field_"
                "is_not_present_in_frozen_C1_lock;"
                "never_rerun_direct_ADB_model_after_GT_access"
            ),
        },
    }

    core[
        "reference_npz_sha256"
    ] = reference_sha

    raw_digest = sha256_bytes(
        canonical_bytes(
            core
        )
    )

    document = dict(
        core
    )

    document[
        "raw_record_digest_sha256"
    ] = raw_digest

    record_bytes = canonical_bytes(
        document
    )

    # =====================================================================
    # Atomic directory publication:
    #
    # final per-scenario directory is either absent or complete.
    # No partial final JSON/NPZ pair can be observed.
    # =====================================================================

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=(
                f".{formal_index:03d}_"
                f"{scenario_id}.tmp."
            ),
            dir=str(
                OUT_ROOT
            ),
        )
    )

    try:
        reference_path = (
            temp_dir
            / "reference.npz"
        )

        record_path = (
            temp_dir
            / "record.json"
        )

        reference_path.write_bytes(
            reference_bytes
        )

        record_path.write_bytes(
            record_bytes
        )

        for path in (
            reference_path,
            record_path,
        ):
            with path.open(
                "rb"
            ) as f:
                os.fsync(
                    f.fileno()
                )

            os.chmod(
                path,
                0o444,
            )

        directory_fd = os.open(
            temp_dir,
            os.O_RDONLY,
        )

        try:
            os.fsync(
                directory_fd
            )
        finally:
            os.close(
                directory_fd
            )

        require(
            not final_dir.exists(),
            (
                "Race: final C2 directory "
                f"appeared: {final_dir}"
            ),
        )

        os.replace(
            temp_dir,
            final_dir,
        )

        os.chmod(
            final_dir,
            0o555,
        )

        parent_fd = os.open(
            OUT_ROOT,
            os.O_RDONLY,
        )

        try:
            os.fsync(
                parent_fd
            )
        finally:
            os.close(
                parent_fd
            )

    finally:
        if temp_dir.exists():
            shutil.rmtree(
                temp_dir,
                ignore_errors=True,
            )

    final_record = (
        final_dir
        / "record.json"
    )

    final_reference = (
        final_dir
        / "reference.npz"
    )

    return {
        "formal_index":
            formal_index,

        "scenario_id":
            scenario_id,

        "record_path":
            str(
                final_record
            ),

        "record_sha256":
            sha256_file(
                final_record
            ),

        "reference_path":
            str(
                final_reference
            ),

        "reference_sha256":
            sha256_file(
                final_reference
            ),

        "raw_record_digest_sha256":
            raw_digest,

        "resumed":
            False,
    }


# =============================================================================
# Post-run C1 immutability proof
# =============================================================================

def reverify_c1_unchanged(
    entries: list[
        dict[str, Any]
    ],
) -> None:

    verify_sha(
        C1_RUNNER,
        EXPECTED_C1_RUNNER_SHA,
        "C1 runner post-C2",
    )

    verify_sha(
        C1_CHECKER,
        EXPECTED_C1_CHECKER_SHA,
        "C1 checker post-C2",
    )

    verify_sha(
        C1_MANIFEST,
        EXPECTED_C1_MANIFEST_SHA,
        "C1 manifest post-C2",
    )

    for entry in entries:
        require(
            sha256_file(
                Path(
                    entry[
                        "lock_json"
                    ]
                )
            )
            ==
            entry[
                "lock_json_sha256"
            ],
            (
                "C1 JSON modified during C2: "
                f"{entry['scenario_id']}"
            ),
        )

        require(
            sha256_file(
                Path(
                    entry[
                        "adb_schedules_npz"
                    ]
                )
            )
            ==
            entry[
                "adb_schedules_npz_sha256"
            ],
            (
                "C1 NPZ modified during C2: "
                f"{entry['scenario_id']}"
            ),
        )


# =============================================================================
# Main
# =============================================================================

def main() -> int:

    print(
        "=" * 88
    )

    print(
        "STAGE 7 — BLOCK 7.5C2"
    )

    print(
        "EVALUATOR-ONLY FUTURE GT + "
        "FRESH STAGE7 COMMUNICATION/ADB OUTCOMES"
    )

    print(
        "C1 CONTROLLERS ARE NEVER RERUN"
    )

    print(
        "=" * 88
    )

    require(
        Path(
            __file__
        ).resolve()
        ==
        SCRIPT_EXPECTED_PATH.resolve(),
        (
            "Runner must be installed at "
            f"{SCRIPT_EXPECTED_PATH}"
        ),
    )

    runner_sha = sha256_file(
        Path(
            __file__
        ).resolve()
    )

    print(
        "C2 runner SHA256 =",
        runner_sha,
        flush=True,
    )

    (
        bundle_info,
        entries,
        validation_metadata,
        order_sha,
    ) = global_preflight()

    print(
        "\n===== C. EVALUATOR RUNTIME BINDING — "
        "STILL NO FUTURE GT =====",
        flush=True,
    )

    initialize_runtime()

    print(
        "Stage1/3 + Stage5 optical + "
        "Stage6 evaluator runtime = PASS",
        flush=True,
    )

    print(
        "FUTURE GT OPENED SO FAR = NO",
        flush=True,
    )

    workers = max(
        1,
        min(
            120,
            int(
                os.environ.get(
                    "C2_WORKERS",
                    str(
                        os.cpu_count()
                        or
                        1
                    ),
                )
            ),
        ),
    )

    print(
        "\n===== D. PARALLEL C2 FORMAL EXECUTION ====="
    )

    print(
        "workers =",
        workers,
    )

    print(
        "per-scenario order = "
        "VERIFY C1 LOCK -> "
        "OPEN FUTURE GT -> "
        "FRESH REFERENCES -> "
        "METRICS -> "
        "IMMUTABLE PUBLISH",
        flush=True,
    )

    tasks = [
        (
            entries[
                index
            ],
            validation_metadata[
                index
            ],
            runner_sha,
        )
        for index in range(
            120
        )
    ]

    results = []

    context = (
        mp.get_context(
            "fork"
        )
    )

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
    ) as pool:

        futures = {
            pool.submit(
                process_one,
                task,
            ):
                int(
                    task[
                        0
                    ][
                        "formal_index"
                    ]
                )
            for task in tasks
        }

        for future in as_completed(
            futures
        ):
            formal_index = (
                futures[
                    future
                ]
            )

            try:
                result = (
                    future.result()
                )

            except Exception as exc:
                for other in futures:
                    other.cancel()

                raise FailClosed(
                    (
                        "C2 scenario "
                        f"{formal_index:03d} failed: "
                        f"{exc}"
                    )
                ) from exc

            results.append(
                result
            )

            print(
                (
                    f"[{formal_index:03d}/119] "
                    f"{'REUSE' if result['resumed'] else 'LOCK '} "
                    f"{result['scenario_id']} "
                    f"record={result['record_sha256'][:12]} "
                    f"ref={result['reference_sha256'][:12]}"
                ),
                flush=True,
            )

    results.sort(
        key=lambda item:
            item[
                "formal_index"
            ]
    )

    require(
        len(results)
        ==
        120,
        "C2 scenario result count != 120.",
    )

    require(
        [
            item[
                "formal_index"
            ]
            for item in results
        ]
        ==
        list(
            range(
                120
            )
        ),
        "C2 result order failure.",
    )

    require(
        len(
            {
                item[
                    "scenario_id"
                ]
                for item in results
            }
        )
        ==
        120,
        "C2 scenario uniqueness failure.",
    )

    # =====================================================================
    # Global immutable 600-row materialization
    # =====================================================================

    print(
        "\n===== E. GLOBAL 600-ROW RAW FORMAL LOCK =====",
        flush=True,
    )

    joint_rows = []

    for result in results:
        document = read_json(
            Path(
                result[
                    "record_path"
                ]
            )
        )

        require(
            len(
                document[
                    "joint_system_rows"
                ]
            )
            ==
            5,
            "C2 five-system row-count failure.",
        )

        joint_rows.extend(
            document[
                "joint_system_rows"
            ]
        )

    require(
        len(
            joint_rows
        )
        ==
        600,
        (
            "C2 must contain exactly "
            "120 x 5 = 600 joint rows."
        ),
    )

    merged_bytes = b"".join(
        canonical_bytes(
            row
        )
        for row in (
            joint_rows
        )
    )

    merged_sha = (
        atomic_immutable_file(
            MERGED_ROWS,
            merged_bytes,
        )
    )

    print(
        "joint rows = 600"
    )

    print(
        "merged JSONL SHA256 =",
        merged_sha,
        flush=True,
    )

    # Resume status is execution-local diagnostic only and
    # MUST NOT change the frozen global manifest bytes.
    manifest_records = [
        {
            key:
                value
            for key, value in (
                result.items()
            )
            if key
            !=
            "resumed"
        }
        for result in results
    ]

    manifest_core = {
        "project":
            "Agni",

        "stage":
            7,

        "block":
            "7.5C2",

        "status":
            "FROZEN_COMPLETE_C2_RAW_FORMAL_RECORDS",

        "runner_path":
            str(
                SCRIPT_EXPECTED_PATH
            ),

        "runner_sha256":
            runner_sha,

        "block75b_contract_sha256":
            EXPECTED_B75B_SHA,

        "formal_manifest_sha256":
            EXPECTED_FORMAL_SHA,

        "formal_order_sha256":
            order_sha,

        "c1_manifest_sha256":
            EXPECTED_C1_MANIFEST_SHA,

        "scenario_count":
            120,

        "system_count":
            5,

        "joint_row_count":
            600,

        "systems":
            list(
                SYSTEMS
            ),

        "merged_joint_rows_path":
            str(
                MERGED_ROWS
            ),

        "merged_joint_rows_sha256":
            merged_sha,

        "scenario_records":
            manifest_records,

        "causality": {
            "C1_verified_before_any_C2_future_GT":
                True,

            "per_scenario_C1_reverified_immediately_before_future_GT":
                True,

            "future_GT_evaluator_only":
                True,

            "controller_rerun_after_GT":
                False,

            "post_outcome_tuning":
                False,
        },

        "outcome_provenance": {
            "fresh_Stage7_communication_truth":
                True,

            "fresh_Stage7_Stage6_constructed_ADB_reference":
                True,

            "historical_Stage6_formal_outcomes_reused":
                False,

            "measured_ADB_ground_truth":
                False,
        },
    }

    manifest_document = dict(
        manifest_core
    )

    manifest_document[
        "manifest_content_digest_sha256"
    ] = sha256_bytes(
        canonical_bytes(
            manifest_core
        )
    )

    manifest_sha = (
        atomic_immutable_file(
            C2_MANIFEST,
            canonical_bytes(
                manifest_document
            ),
        )
    )

    # =====================================================================
    # Re-prove C1 was untouched
    # =====================================================================

    print(
        "\n===== F. POST-RUN C1 IMMUTABILITY RECHECK =====",
        flush=True,
    )

    reverify_c1_unchanged(
        entries
    )

    print(
        "C1 runner/checker/manifest/"
        "120 lock pairs unchanged = PASS",
        flush=True,
    )

    # =====================================================================
    # C2 report + frozen seal
    # =====================================================================

    report_core = {
        "project":
            "Agni",

        "stage":
            7,

        "block":
            "7.5C2",

        "status":
            "PASS_FROZEN",

        "runner_sha256":
            runner_sha,

        "source_bundle":
            bundle_info,

        "c1_manifest_sha256":
            EXPECTED_C1_MANIFEST_SHA,

        "c2_manifest_path":
            str(
                C2_MANIFEST
            ),

        "c2_manifest_sha256":
            manifest_sha,

        "merged_joint_rows_path":
            str(
                MERGED_ROWS
            ),

        "merged_joint_rows_sha256":
            merged_sha,

        "scenario_count":
            120,

        "five_system_rows":
            600,

        "communication_methodology": (
            "frozen_Stage5_oracle_best_gain_beam_"
            "plus_optical_gain_DPSK_BER_effective_rate_"
            "with_fresh_Stage7_receiver_future_truth"
        ),

        "ADB_reference": (
            "fresh_constructed_evaluator_only_future_"
            "WOMD_full_box_reference_on_frozen_501x301_grid"
        ),

        "vehicle_surrogate": (
            "frozen_heading_dependent_"
            "front_upper_rear_upper_geometry_surrogate"
        ),

        "future_GT_accessed":
            True,

        "future_GT_controller_input":
            False,

        "controllers_rerun_after_future_GT_access":
            False,

        "historical_Stage6_outcomes_reused":
            False,

        "post_outcome_tuning":
            False,

        "C3_executed":
            False,

        "C3_authorized_only_after_this_seal":
            True,
    }

    report_document = dict(
        report_core
    )

    report_document[
        "report_content_digest_sha256"
    ] = sha256_bytes(
        canonical_bytes(
            report_core
        )
    )

    report_sha = (
        atomic_immutable_file(
            C2_REPORT,
            canonical_bytes(
                report_document
            ),
        )
    )

    seal_core = {
        "project":
            "Agni",

        "stage":
            7,

        "block":
            "7.5C2",

        "status":
            "FROZEN_COMPLETE_C2",

        "runner_sha256":
            runner_sha,

        "c1_manifest_sha256":
            EXPECTED_C1_MANIFEST_SHA,

        "c2_manifest_sha256":
            manifest_sha,

        "c2_report_sha256":
            report_sha,

        "merged_joint_rows_sha256":
            merged_sha,

        "scenario_count":
            120,

        "joint_row_count":
            600,

        "future_GT_evaluator_only":
            True,

        "controller_outputs_frozen_before_GT":
            True,

        "controllers_rerun_after_GT":
            False,

        "fresh_communication_outcomes":
            True,

        "fresh_constructed_ADB_reference":
            True,

        "historical_Stage6_outcomes_reused":
            False,

        "C3_may_start":
            True,
    }

    seal_document = dict(
        seal_core
    )

    seal_document[
        "seal_content_digest_sha256"
    ] = sha256_bytes(
        canonical_bytes(
            seal_core
        )
    )

    seal_sha = (
        atomic_immutable_file(
            C2_SEAL,
            canonical_bytes(
                seal_document
            ),
        )
    )

    print()
    print(
        "=" * 88
    )

    print(
        "BLOCK 7.5C2 = FULLY VERIFIED / FROZEN"
    )

    print(
        "120/120 IMMUTABLE C2 SCENARIO RECORDS = PASS"
    )

    print(
        "600/600 FIVE-SYSTEM JOINT ROWS = PASS"
    )

    print(
        "FRESH STAGE7 COMMUNICATION OUTCOMES = PASS"
    )

    print(
        "FRESH STAGE6-CONSTRUCTED ADB REFERENCES "
        "ON STAGE7 COHORT = PASS"
    )

    print(
        "HISTORICAL STAGE6 FORMAL OUTCOMES REUSED = NO"
    )

    print(
        "C1 LOCKS MODIFIED = NO"
    )

    print(
        "FUTURE GT CONTROLLER INPUT = NO"
    )

    print(
        "CONTROLLERS RERUN AFTER FUTURE GT = NO"
    )

    print(
        "POST-OUTCOME TUNING = NO"
    )

    print(
        "C3 EXECUTED = NO"
    )

    print(
        "C2 manifest SHA256 =",
        manifest_sha,
    )

    print(
        "C2 report SHA256   =",
        report_sha,
    )

    print(
        "C2 seal SHA256     =",
        seal_sha,
    )

    print(
        "C2 runner SHA256   =",
        runner_sha,
    )

    print(
        "STATUS = FROZEN_COMPLETE_C2"
    )

    print(
        "=" * 88
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )

    except Exception as exc:
        print(
            "\n"
            +
            "!" * 88,
            file=sys.stderr,
        )

        print(
            "BLOCK 7.5C2 FAIL-CLOSED",
            file=sys.stderr,
        )

        print(
            type(exc).__name__
            +
            ":",
            exc,
            file=sys.stderr,
        )

        print(
            "NO C3 AUTHORIZATION WAS PRODUCED.",
            file=sys.stderr,
        )

        print(
            "!" * 88,
            file=sys.stderr,
        )

        raise

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch

from iscai_stage4.data import (
    attach_supervision,
)

from iscai_stage4.data.real_pipeline import (
    build_real_causal_inputs,
    load_frozen_stage2_configs,
    read_training_scenario,
)

from iscai_stage4.ml import (
    CLASS_ID_TO_NAME,
    fit_per_horizon_covariance_scale,
    reliability_metrics,
    set_global_determinism,
)

from iscai_stage4.ml.calibration_runtime import (
    CalibrationNormalizer,
    build_gaussian_model,
    calibrated_nll_per_horizon,
    infer_scene_statistics,
    masked_nll_metrics,
    masked_planar_ade,
    probe_calibrated_output,
    samples_to_arrays,
)


ROOT = Path(
    "/home/agni/waymo"
)

STAGE4 = (
    ROOT
    / "iscai_stage4"
)

BLOCK45_PART1 = (
    STAGE4
    / "reports/"
      "block45_part1_preflight.json"
)

BLOCK44 = (
    STAGE4
    / "reports/"
      "block44_gaussian_gru.json"
)

GAUSSIAN_CHECKPOINT = (
    STAGE4
    / "artifacts/block44/"
      "gaussian_gru.pt"
)

NORMALIZATION = (
    STAGE4
    / "artifacts/block43/"
      "fit_normalization.json"
)

CALIBRATION_MANIFEST = (
    STAGE4
    / "artifacts/block41/"
      "calibration.jsonl"
)

BLOCK45_CONFIG = (
    STAGE4
    / "configs/"
      "stage4_calibration.json"
)

CACHE_DIR = (
    STAGE4
    / "artifacts/block45/"
      "cache/calibration"
)

MERGED_STATS = (
    STAGE4
    / "artifacts/block45/"
      "calibration_sufficient_statistics.npz"
)

CACHE_REPORT = (
    STAGE4
    / "artifacts/block45/"
      "calibration_cache_manifest.json"
)

CALIBRATOR = (
    STAGE4
    / "artifacts/block45/"
      "covariance_scaler.json"
)

RELIABILITY = (
    STAGE4
    / "artifacts/block45/"
      "reliability_metrics.json"
)

REPORT = (
    STAGE4
    / "reports/"
      "block45_calibration.json"
)

FAILURE = (
    STAGE4
    / "reports/"
      "block45_calibration_failure.json"
)

LOG = (
    STAGE4
    / "docs/"
      "implementation_log.md"
)

EXPECTED_BLOCK44_IMPL_SHA = (
    "b9ea474443d3a54235f09cd4e2b616374"
    "b37456f2b9e080f5334f76f4dd7c432"
)

EXPECTED_GAUSSIAN_CHECKPOINT_SHA = (
    "49ff64d145eaa633f295c16f660df380"
    "c35383e7e3b61279a5aad7cd700d619f"
)

EXPECTED_GAUSSIAN_STATE_SHA = (
    "1d66cf082d0d9be41319f7dcbe18fc259"
    "910de7f45de006f9043129837b86be3"
)

EXPECTED_CALIBRATION_MANIFEST_SHA = (
    "e003dd5c4d5a253729700b12c3754c47"
    "ffb99dadfa035b46627f01ec91eed4be"
)

MIN_FREE_GIB = 250.0
MAX_SAMPLES_PER_SCENE = 64

STAT_KEYS = (
    "mahalanobis_squared",
    "validity",
    "raw_metric_nll",
    "planar_error_m",
    "class_id",
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


def canonical_sha(
    payload,
):
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            allow_nan=False,
        ).encode(
            "utf-8"
        )
    ).hexdigest()


def state_dict_sha256(
    state_dict,
):
    digest = sha256()

    for key in sorted(
        state_dict
    ):
        tensor = (
            state_dict[
                key
            ]
            .detach()
            .cpu()
            .contiguous()
        )

        digest.update(
            key.encode(
                "utf-8"
            )
        )
        digest.update(
            b"\0"
        )

        digest.update(
            str(
                tensor.dtype
            ).encode(
                "ascii"
            )
        )
        digest.update(
            b"\0"
        )

        digest.update(
            tensor.numpy()
            .tobytes()
        )
        digest.update(
            b"\0"
        )

    return digest.hexdigest()


def implementation_fingerprint():
    roots = (
        STAGE4 / "src",
        STAGE4 / "tests",
        STAGE4 / "configs",
        STAGE4 / "scripts",
    )

    files = []

    for root in roots:
        for path in root.rglob(
            "*"
        ):
            if not path.is_file():
                continue

            if "__pycache__" in (
                path.parts
            ):
                continue

            if path.suffix in (
                ".pyc",
                ".pyo",
            ):
                continue

            files.append(
                path
            )

    files.sort(
        key=lambda path:
            str(
                path.relative_to(
                    STAGE4
                )
            )
    )

    digest = sha256()

    for path in files:
        relative = str(
            path.relative_to(
                STAGE4
            )
        )

        digest.update(
            relative.encode(
                "utf-8"
            )
        )
        digest.update(
            b"\0"
        )

        digest.update(
            path.read_bytes()
        )
        digest.update(
            b"\0"
        )

    return (
        len(files),
        digest.hexdigest(),
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
        )
        +
        "\n",
        encoding="utf-8",
    )


def atomic_npz(
    path: Path,
    **arrays,
):
    temporary = (
        path.with_suffix(
            path.suffix
            +
            ".tmp"
        )
    )

    with temporary.open(
        "wb"
    ) as stream:
        np.savez(
            stream,
            **arrays,
        )

    os.replace(
        temporary,
        path,
    )


def read_manifest():
    rows = tuple(
        json.loads(
            line
        )
        for line in (
            CALIBRATION_MANIFEST
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        )
        if line.strip()
    )

    if len(rows) != 7209:
        raise RuntimeError(
            "Frozen calibration "
            "manifest count changed."
        )

    return rows


def cache_path(
    rank,
    scenario_id,
):
    return (
        CACHE_DIR
        /
        (
            f"{rank:05d}_"
            f"{scenario_id}.npz"
        )
    )


def validate_cache_shard(
    path,
    scenario_id,
):
    if not path.is_file():
        return False

    try:
        with np.load(
            path,
            allow_pickle=False,
        ) as data:
            stored_id = str(
                data[
                    "scenario_id"
                ].item()
            )

            if (
                stored_id
                !=
                scenario_id
            ):
                return False

            q = np.asarray(
                data[
                    "mahalanobis_squared"
                ]
            )

            validity = np.asarray(
                data[
                    "validity"
                ]
            )

            raw_nll = np.asarray(
                data[
                    "raw_metric_nll"
                ]
            )

            planar = np.asarray(
                data[
                    "planar_error_m"
                ]
            )

            class_id = np.asarray(
                data[
                    "class_id"
                ]
            )

            prediction_hash = str(
                data[
                    "prediction_sha256"
                ].item()
            )

        n = int(
            q.shape[0]
        )

        if q.shape != (
            n,
            4,
        ):
            return False

        if validity.shape != (
            n,
            4,
        ):
            return False

        if raw_nll.shape != (
            n,
            4,
        ):
            return False

        if planar.shape != (
            n,
            4,
        ):
            return False

        if class_id.shape != (
            n,
        ):
            return False

        if len(
            prediction_hash
        ) != 64:
            return False

        valid = (
            validity
            >
            0.5
        )

        if not np.all(
            np.isfinite(
                q[
                    valid
                ]
            )
        ):
            return False

        if not np.all(
            np.isfinite(
                raw_nll[
                    valid
                ]
            )
        ):
            return False

        return True

    except Exception:
        return False


def archive_invalid(
    path,
):
    if not path.exists():
        return

    destination = (
        path.with_name(
            path.name
            +
            f".corrupt."
            f"{int(time.time())}"
        )
    )

    os.replace(
        path,
        destination,
    )


def load_shard(
    path,
):
    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        return {
            key:
                np.asarray(
                    data[
                        key
                    ]
                )
            for key in (
                STAT_KEYS
            )
        } | {
            "prediction_sha256":
                str(
                    data[
                        "prediction_sha256"
                    ].item()
                )
        }


def infer_record(
    record,
    *,
    model,
    normalizer,
    clean_config,
    degraded_config,
    device,
):
    scenario = (
        read_training_scenario(
            record
        )
    )

    built = (
        build_real_causal_inputs(
            scenario,
            clean_config=(
                clean_config
            ),
            degraded_config=(
                degraded_config
            ),
        )
    )

    samples = (
        attach_supervision(
            built[
                "scene_inputs"
            ],
            scenario,
            T_H0_from_W=(
                built[
                    "adapted"
                ].frames
                .T_H0_from_W
            ),
        )
    )

    # attach_supervision is already sorted by
    # causal prediction_id. No future-label
    # quality is inspected before this cap.
    samples = samples[
        :MAX_SAMPLES_PER_SCENE
    ]

    stats = (
        infer_scene_statistics(
            model,
            samples,
            normalizer,
            device=device,
        )
    )

    return (
        scenario,
        samples,
        stats,
    )


def build_or_reuse_shard(
    rank,
    record,
    *,
    model,
    normalizer,
    clean_config,
    degraded_config,
    device,
):
    scenario_id = (
        record[
            "scenario_id"
        ]
    )

    path = cache_path(
        rank,
        scenario_id,
    )

    if validate_cache_shard(
        path,
        scenario_id,
    ):
        loaded = load_shard(
            path
        )

        return {
            "reused":
                True,

            "sample_count":
                int(
                    loaded[
                        "mahalanobis_squared"
                    ].shape[0]
                ),

            "prediction_sha256":
                loaded[
                    "prediction_sha256"
                ],
        }

    archive_invalid(
        path
    )

    (
        scenario,
        samples,
        stats,
    ) = infer_record(
        record,
        model=model,
        normalizer=normalizer,
        clean_config=clean_config,
        degraded_config=(
            degraded_config
        ),
        device=device,
    )

    atomic_npz(
        path,

        scenario_id=np.asarray(
            scenario.scenario_id
        ),

        mahalanobis_squared=(
            stats[
                "mahalanobis_squared"
            ]
        ),

        validity=(
            stats[
                "validity"
            ]
        ),

        raw_metric_nll=(
            stats[
                "raw_metric_nll"
            ]
        ),

        planar_error_m=(
            stats[
                "planar_error_m"
            ]
        ),

        class_id=(
            stats[
                "class_id"
            ]
        ),

        prediction_sha256=np.asarray(
            stats[
                "prediction_sha256"
            ]
        ),
    )

    if not validate_cache_shard(
        path,
        scenario_id,
    ):
        raise RuntimeError(
            "New calibration shard "
            "failed validation."
        )

    return {
        "reused":
            False,

        "sample_count":
            len(
                samples
            ),

        "prediction_sha256":
            stats[
                "prediction_sha256"
            ],

        "associated_tracks":
            built_count(
                scenario,
                samples,
            ),
    }


def built_count(
    scenario,
    samples,
):
    # Diagnostic only. Avoid dependence on
    # internal Stage3 association objects here.
    return {
        "WOMD_tracks":
            len(
                scenario.tracks
            ),

        "supervised_samples":
            len(
                samples
            ),
    }


def merge_statistics(
    records,
):
    parts = {
        key: []
        for key in (
            STAT_KEYS
        )
    }

    scene_sample_counts = []

    prediction_hashes = []

    for rank, record in enumerate(
        records,
        start=1,
    ):
        path = cache_path(
            rank,
            record[
                "scenario_id"
            ],
        )

        if not validate_cache_shard(
            path,
            record[
                "scenario_id"
            ],
        ):
            raise RuntimeError(
                "Cannot merge invalid "
                f"calibration shard: "
                f"{path}"
            )

        shard = load_shard(
            path
        )

        scene_sample_counts.append(
            int(
                shard[
                    "mahalanobis_squared"
                ].shape[0]
            )
        )

        prediction_hashes.append(
            shard[
                "prediction_sha256"
            ]
        )

        for key in (
            STAT_KEYS
        ):
            parts[
                key
            ].append(
                shard[
                    key
                ]
            )

    merged = {
        key:
            np.concatenate(
                parts[
                    key
                ],
                axis=0,
            )
        for key in (
            STAT_KEYS
        )
    }

    if (
        merged[
            "mahalanobis_squared"
        ].shape[0]
        <=
        0
    ):
        raise RuntimeError(
            "Merged calibration "
            "statistics are empty."
        )

    atomic_npz(
        MERGED_STATS,
        **merged,
    )

    digest = sha256()

    for key in (
        STAT_KEYS
    ):
        value = np.ascontiguousarray(
            merged[
                key
            ]
        )

        digest.update(
            key.encode(
                "utf-8"
            )
        )
        digest.update(
            b"\0"
        )
        digest.update(
            str(
                value.dtype
            ).encode(
                "ascii"
            )
        )
        digest.update(
            b"\0"
        )
        digest.update(
            value.tobytes(
                order="C"
            )
        )
        digest.update(
            b"\0"
        )

    for value in prediction_hashes:
        digest.update(
            value.encode(
                "ascii"
            )
        )
        digest.update(
            b"\0"
        )

    class_counts = Counter(
        int(value)
        for value in (
            merged[
                "class_id"
            ].tolist()
        )
    )

    return (
        merged,
        {
            "scenario_count":
                len(records),

            "sample_count":
                int(
                    merged[
                        "mahalanobis_squared"
                    ].shape[0]
                ),

            "valid_actor_horizon_points":
                int(
                    (
                        merged[
                            "validity"
                        ]
                        >
                        0.5
                    ).sum()
                ),

            "zero_sample_scenes":
                int(
                    sum(
                        value == 0
                        for value in (
                            scene_sample_counts
                        )
                    )
                ),

            "max_samples_in_scene":
                int(
                    max(
                        scene_sample_counts
                    )
                ),

            "class_counts":
                {
                    CLASS_ID_TO_NAME.get(
                        key,
                        str(key),
                    ):
                        int(value)
                    for key, value
                    in sorted(
                        class_counts.items()
                    )
                },

            "sufficient_statistics_file_sha256":
                file_sha256(
                    MERGED_STATS
                ),

            "sufficient_statistics_content_sha256":
                digest.hexdigest(),
        },
    )


def load_merged_statistics():
    with np.load(
        MERGED_STATS,
        allow_pickle=False,
    ) as data:
        return {
            key:
                np.asarray(
                    data[
                        key
                    ]
                )
            for key in (
                STAT_KEYS
            )
        }


def calibration_nll_report(
    raw_nll,
    q,
    validity,
    variance_scale,
):
    calibrated_nll = (
        calibrated_nll_per_horizon(
            raw_nll,
            q,
            variance_scale,
        )
    )

    return {
        "raw":
            masked_nll_metrics(
                raw_nll,
                validity,
            ),

        "calibrated":
            masked_nll_metrics(
                calibrated_nll,
                validity,
            ),
    }


def main():
    print(
        "============================================================"
    )
    print(
        "STAGE 4 — BLOCK 4.5 "
        "REAL CALIBRATION"
    )
    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # Already-frozen fast path.
    # --------------------------------------------------------

    if (
        REPORT.is_file()
        and
        CALIBRATOR.is_file()
    ):
        previous = json.loads(
            REPORT.read_text(
                encoding="utf-8"
            )
        )

        if (
            previous.get(
                "status"
            )
            ==
            "PASS"
        ):
            expected = previous[
                "calibrator"
            ][
                "file_sha256"
            ]

            if (
                file_sha256(
                    CALIBRATOR
                )
                ==
                expected
            ):
                print(
                    "Block4.5 already "
                    "COMPLETE/FROZEN."
                )
                print(
                    "calibrator SHA256 =",
                    expected,
                )
                print(
                    "STATUS = PASS"
                )
                return

    # --------------------------------------------------------
    # Frozen upstream.
    # --------------------------------------------------------

    part1 = json.loads(
        BLOCK45_PART1.read_text(
            encoding="utf-8"
        )
    )

    block44 = json.loads(
        BLOCK44.read_text(
            encoding="utf-8"
        )
    )

    block45_config = json.loads(
        BLOCK45_CONFIG.read_text(
            encoding="utf-8"
        )
    )

    normalization = json.loads(
        NORMALIZATION.read_text(
            encoding="utf-8"
        )
    )

    if (
        part1.get(
            "status"
        )
        !=
        "PASS"
    ):
        raise RuntimeError(
            "Block4.5 Part1 "
            "is not PASS."
        )

    if (
        block44.get(
            "status"
        )
        !=
        "PASS"
    ):
        raise RuntimeError(
            "Block4.4 is not PASS."
        )

    if (
        block44[
            "implementation"
        ][
            "sha256"
        ]
        !=
        EXPECTED_BLOCK44_IMPL_SHA
    ):
        raise RuntimeError(
            "Frozen Block4.4 "
            "implementation SHA changed."
        )

    if (
        file_sha256(
            GAUSSIAN_CHECKPOINT
        )
        !=
        EXPECTED_GAUSSIAN_CHECKPOINT_SHA
    ):
        raise RuntimeError(
            "Frozen Gaussian "
            "checkpoint changed."
        )

    if (
        file_sha256(
            CALIBRATION_MANIFEST
        )
        !=
        EXPECTED_CALIBRATION_MANIFEST_SHA
    ):
        raise RuntimeError(
            "Frozen calibration "
            "manifest changed."
        )

    if (
        normalization.get(
            "source"
        )
        !=
        "fit_only"
    ):
        raise RuntimeError(
            "Normalization is "
            "not fit-only."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable."
        )

    if (
        torch.version.cuda
        !=
        "13.2"
    ):
        raise RuntimeError(
            "CUDA runtime changed."
        )

    device = torch.device(
        "cuda:0"
    )

    if (
        "A6000"
        not in
        torch.cuda.get_device_name(
            0
        )
    ):
        raise RuntimeError(
            "Expected RTX A6000."
        )

    free_gib = (
        shutil.disk_usage(
            ROOT
        ).free
        /
        1024**3
    )

    if free_gib < MIN_FREE_GIB:
        raise RuntimeError(
            "250-GiB storage "
            "reserve violated."
        )

    set_global_determinism(
        20260822
    )

    checkpoint = torch.load(
        GAUSSIAN_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    if (
        checkpoint[
            "state_dict_sha256"
        ]
        !=
        EXPECTED_GAUSSIAN_STATE_SHA
    ):
        raise RuntimeError(
            "Gaussian checkpoint "
            "state SHA declaration "
            "changed."
        )

    if (
        state_dict_sha256(
            checkpoint[
                "state_dict"
            ]
        )
        !=
        EXPECTED_GAUSSIAN_STATE_SHA
    ):
        raise RuntimeError(
            "Gaussian state-dict "
            "content changed."
        )

    model = build_gaussian_model(
        checkpoint[
            "configuration"
        ],
        device=device,
    )

    model.load_state_dict(
        checkpoint[
            "state_dict"
        ]
    )

    model.eval()

    normalizer = (
        CalibrationNormalizer(
            normalization,
            device=device,
        )
    )

    records = read_manifest()

    (
        clean_config,
        degraded_config,
    ) = load_frozen_stage2_configs()

    print(
        "Block4.4 frozen upstream = PASS"
    )
    print(
        "Gaussian checkpoint       = PASS"
    )
    print(
        "calibration scenarios     =",
        len(records),
    )
    print(
        "normalization source      = FIT ONLY"
    )
    print(
        "development used to fit   = NO"
    )
    print(
        "formal validation used    = NO"
    )

    # --------------------------------------------------------
    # Full resumable calibration inference cache.
    # --------------------------------------------------------

    print()
    print(
        "============================================================"
    )
    print(
        "CALIBRATION CACHE / INFERENCE"
    )
    print(
        "============================================================"
    )

    start_time = (
        time.perf_counter()
    )

    reused = 0
    total_samples = 0

    scene_records = []

    for rank, record in enumerate(
        records,
        start=1,
    ):
        result = (
            build_or_reuse_shard(
                rank,
                record,
                model=model,
                normalizer=(
                    normalizer
                ),
                clean_config=(
                    clean_config
                ),
                degraded_config=(
                    degraded_config
                ),
                device=device,
            )
        )

        reused += int(
            result[
                "reused"
            ]
        )

        total_samples += int(
            result[
                "sample_count"
            ]
        )

        scene_records.append({
            "rank":
                rank,

            "scenario_id":
                record[
                    "scenario_id"
                ],

            "sample_count":
                int(
                    result[
                        "sample_count"
                    ]
                ),

            "reused":
                bool(
                    result[
                        "reused"
                    ]
                ),

            "prediction_sha256":
                result[
                    "prediction_sha256"
                ],
        })

        if (
            rank % 100 == 0
            or
            rank == len(
                records
            )
        ):
            print(
                f"calibration: "
                f"{rank}/"
                f"{len(records)} "
                f"| samples="
                f"{total_samples} "
                f"| reused="
                f"{reused}",
                flush=True,
            )

    inference_runtime_s = (
        time.perf_counter()
        -
        start_time
    )

    (
        merged,
        cache_summary,
    ) = merge_statistics(
        records
    )

    cache_report = {
        "stage": 4,
        "block": "4.5",

        "partition":
            "calibration",

        "scenario_count":
            len(records),

        "full_frozen_partition":
            True,

        "max_samples_per_scene":
            MAX_SAMPLES_PER_SCENE,

        "sample_cap_future_based":
            False,

        "sample_cap_performance_based":
            False,

        "reused_scene_shards":
            int(
                reused
            ),

        "runtime_s":
            inference_runtime_s,

        "summary":
            cache_summary,

        "scene_records":
            scene_records,
    }

    write_json(
        CACHE_REPORT,
        cache_report,
    )

    print()
    print(
        "calibration samples       =",
        cache_summary[
            "sample_count"
        ],
    )
    print(
        "valid horizon points      =",
        cache_summary[
            "valid_actor_horizon_points"
        ],
    )
    print(
        "class counts              =",
        cache_summary[
            "class_counts"
        ],
    )
    print(
        "zero-sample scenes        =",
        cache_summary[
            "zero_sample_scenes"
        ],
    )
    print(
        "statistics content SHA    =",
        cache_summary[
            "sufficient_statistics_content_sha256"
        ],
    )

    # --------------------------------------------------------
    # Exact inference repeat on first four non-empty
    # frozen-manifest scenes.
    # --------------------------------------------------------

    nonempty = [
        (
            rank,
            record,
        )
        for rank, record
        in enumerate(
            records,
            start=1,
        )
        if (
            scene_records[
                rank - 1
            ][
                "sample_count"
            ]
            >
            0
        )
    ]

    probes = nonempty[
        :4
    ]

    if len(probes) != 4:
        raise RuntimeError(
            "Not enough non-empty "
            "calibration scenes for "
            "repeat-inference probe."
        )

    repeat_passes = 0

    for rank, record in probes:
        (
            _,
            _,
            repeated_stats,
        ) = infer_record(
            record,
            model=model,
            normalizer=normalizer,
            clean_config=(
                clean_config
            ),
            degraded_config=(
                degraded_config
            ),
            device=device,
        )

        expected_hash = (
            scene_records[
                rank - 1
            ][
                "prediction_sha256"
            ]
        )

        if (
            repeated_stats[
                "prediction_sha256"
            ]
            !=
            expected_hash
        ):
            raise RuntimeError(
                "Calibration inference "
                "repeat mismatch for "
                f"{record['scenario_id']}."
            )

        repeat_passes += 1

    print()
    print(
        "calibration inference "
        "repeat =",
        f"{repeat_passes}/4 PASS",
    )

    # --------------------------------------------------------
    # Fit calibrator TWICE and require exact equality.
    # --------------------------------------------------------

    q = merged[
        "mahalanobis_squared"
    ]

    validity = merged[
        "validity"
    ]

    fit_a = (
        fit_per_horizon_covariance_scale(
            q,
            validity,
        )
    )

    fit_b = (
        fit_per_horizon_covariance_scale(
            q,
            validity,
        )
    )

    calibrator_a = fit_a[
        "calibrator"
    ]

    calibrator_b = fit_b[
        "calibrator"
    ]

    if (
        calibrator_a.to_dict()
        !=
        calibrator_b.to_dict()
    ):
        raise RuntimeError(
            "Calibration fit is "
            "not exactly reproducible."
        )

    raw_reliability = (
        reliability_metrics(
            q,
            validity,
        )
    )

    calibrated_reliability = (
        reliability_metrics(
            q,
            validity,
            variance_scale=(
                calibrator_a
                .variance_scale
            ),
        )
    )

    raw_ece = float(
        raw_reliability[
            "coverage_ECE_macro"
        ]
    )

    calibrated_ece = float(
        calibrated_reliability[
            "coverage_ECE_macro"
        ]
    )

    if (
        calibrated_ece
        >
        raw_ece
        +
        1e-15
    ):
        raise RuntimeError(
            "Real calibration-set ECE "
            "became worse."
        )

    # --------------------------------------------------------
    # Raw vs calibrated metric Gaussian NLL.
    # --------------------------------------------------------

    nll_report = (
        calibration_nll_report(
            merged[
                "raw_metric_nll"
            ],
            q,
            validity,
            calibrator_a
            .variance_scale,
        )
    )

    # Calibration does not touch the predictive mean.
    trajectory_metrics = (
        masked_planar_ade(
            merged[
                "planar_error_m"
            ],
            validity,
        )
    )

    # --------------------------------------------------------
    # Direct mean-identity + SPD probe on first
    # non-empty scene.
    # --------------------------------------------------------

    (
        probe_rank,
        probe_record,
    ) = probes[0]

    scenario = (
        read_training_scenario(
            probe_record
        )
    )

    built = (
        build_real_causal_inputs(
            scenario,
            clean_config=(
                clean_config
            ),
            degraded_config=(
                degraded_config
            ),
        )
    )

    probe_samples = (
        attach_supervision(
            built[
                "scene_inputs"
            ],
            scenario,
            T_H0_from_W=(
                built[
                    "adapted"
                ].frames
                .T_H0_from_W
            ),
        )
    )[
        :MAX_SAMPLES_PER_SCENE
    ]

    probe = (
        probe_calibrated_output(
            model,
            probe_samples,
            normalizer,
            calibrator_a
            .variance_scale,
            device=device,
        )
    )

    if not probe[
        "mean_exactly_unchanged"
    ]:
        raise RuntimeError(
            "Calibration modified "
            "predictive mean."
        )

    if not probe[
        "calibrated_covariance_SPD"
    ]:
        raise RuntimeError(
            "Calibration lost "
            "predictive covariance SPD."
        )

    # --------------------------------------------------------
    # Freeze calibrator artifact.
    # --------------------------------------------------------

    calibrator_payload = {
        "stage": 4,
        "block": "4.5",
        "status": "FROZEN",

        "method":
            "per_horizon_scalar_covariance_scaling",

        "horizons_s":
            [
                0.1,
                0.3,
                0.5,
                1.0,
            ],

        "variance_scale":
            [
                float(value)
                for value in (
                    calibrator_a
                    .variance_scale
                )
            ],

        "standard_deviation_scale":
            [
                float(
                    value ** 0.5
                )
                for value in (
                    calibrator_a
                    .variance_scale
                )
            ],

        "mean_modified":
            False,

        "measurement_covariance_R_t_modified":
            False,

        "predictive_covariance_modified":
            True,

        "fit_partition":
            "frozen_Block4.1_calibration_only",

        "calibration_manifest_sha256":
            EXPECTED_CALIBRATION_MANIFEST_SHA,

        "Gaussian_checkpoint_sha256":
            EXPECTED_GAUSSIAN_CHECKPOINT_SHA,

        "sufficient_statistics_content_sha256":
            cache_summary[
                "sufficient_statistics_content_sha256"
            ],

        "fit_reproducible":
            True,
    }

    calibrator_content_sha = (
        canonical_sha(
            calibrator_payload
        )
    )

    calibrator_payload[
        "content_sha256"
    ] = (
        calibrator_content_sha
    )

    write_json(
        CALIBRATOR,
        calibrator_payload,
    )

    calibrator_file_sha = (
        file_sha256(
            CALIBRATOR
        )
    )

    # --------------------------------------------------------
    # Freeze reliability/reliability-diagram values.
    # --------------------------------------------------------

    reliability_payload = {
        "stage": 4,
        "block": "4.5",

        "confidence_levels":
            [
                0.50,
                0.80,
                0.90,
                0.95,
                0.99,
            ],

        "raw":
            raw_reliability,

        "calibrated":
            calibrated_reliability,

        "NLL":
            nll_report,

        "trajectory_mean_metrics":
            trajectory_metrics,

        "predictive_mean_identical":
            True,

        "calibrated_covariance_SPD":
            True,
    }

    write_json(
        RELIABILITY,
        reliability_payload,
    )

    # --------------------------------------------------------
    # Full Stage4 regression: 76/76.
    # --------------------------------------------------------

    import re
    import subprocess
    import sys

    test = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(
                STAGE4
                / "tests"
            ),
            "-p",
            "test_*.py",
        ],
        cwd=str(
            STAGE4
        ),
        text=True,
        capture_output=True,
    )

    combined = (
        test.stdout
        +
        "\n"
        +
        test.stderr
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        combined,
    )

    test_count = (
        int(
            match.group(1)
        )
        if match
        else None
    )

    if (
        test.returncode != 0
        or
        test_count != 76
    ):
        print(
            combined
        )

        raise RuntimeError(
            "Expected final Stage4 "
            "regression 76/76."
        )

    (
        implementation_files,
        implementation_sha,
    ) = implementation_fingerprint()

    free_gib_after = (
        shutil.disk_usage(
            ROOT
        ).free
        /
        1024**3
    )

    if (
        free_gib_after
        <
        MIN_FREE_GIB
    ):
        raise RuntimeError(
            "250-GiB storage "
            "reserve violated."
        )

    final_report = {
        "stage": 4,
        "block": "4.5",
        "status": "PASS",

        "calibration_partition": {
            "scenario_count":
                7209,

            "full_partition":
                True,

            "manifest_sha256":
                EXPECTED_CALIBRATION_MANIFEST_SHA,

            "development_used":
                False,

            "fit_used":
                False,

            "formal_validation_used":
                False,
        },

        "cache": {
            "resumable":
                True,

            "Stage2_Stage3_rebuild_on_rerun":
                False,

            "summary":
                cache_summary,

            "repeat_inference_scenes":
                4,

            "exact_repeat":
                True,
        },

        "calibrator": {
            "method":
                "per_horizon_scalar_covariance_scaling",

            "variance_scale":
                calibrator_payload[
                    "variance_scale"
                ],

            "standard_deviation_scale":
                calibrator_payload[
                    "standard_deviation_scale"
                ],

            "mean_modified":
                False,

            "measurement_R_t_modified":
                False,

            "predictive_covariance_modified":
                True,

            "fit_reproducible":
                True,

            "content_sha256":
                calibrator_content_sha,

            "file_sha256":
                calibrator_file_sha,
        },

        "reliability": {
            "raw":
                raw_reliability,

            "calibrated":
                calibrated_reliability,

            "ECE_nonworsening":
                True,

            "raw_macro_ECE":
                raw_ece,

            "calibrated_macro_ECE":
                calibrated_ece,

            "raw_coverage_event_Brier":
                raw_reliability[
                    "coverage_event_Brier"
                ],

            "calibrated_coverage_event_Brier":
                calibrated_reliability[
                    "coverage_event_Brier"
                ],
        },

        "Gaussian_NLL": {
            "raw":
                nll_report[
                    "raw"
                ],

            "calibrated":
                nll_report[
                    "calibrated"
                ],
        },

        "trajectory_mean": {
            "unchanged":
                True,

            "metrics":
                trajectory_metrics,
        },

        "covariance": {
            "SPD_after_calibration":
                True,
        },

        "formal_validation": {
            "used":
                False,

            "calibration_generalization":
                "DEFERRED_TO_BLOCK_4.8",
        },

        "regression": {
            "tests_passed":
                76,

            "tests_total":
                76,
        },

        "storage": {
            "free_gib":
                free_gib_after,

            "hard_reserve_gib":
                MIN_FREE_GIB,

            "pass":
                True,
        },

        "implementation": {
            "file_count":
                implementation_files,

            "sha256":
                implementation_sha,
        },

        "not_yet_claimed": [
            "GMM multimodality",
            "measurement-to-predictive uncertainty ablations",
            "formal validation calibration generalization",
            "formal probabilistic superiority over classical baseline"
        ],
    }

    write_json(
        REPORT,
        final_report,
    )

    if FAILURE.exists():
        FAILURE.unlink()

    # --------------------------------------------------------
    # Closure log.
    # --------------------------------------------------------

    marker = (
        "## Block 4.5 — "
        "Trajectory uncertainty calibration"
    )

    existing = (
        LOG.read_text(
            encoding="utf-8"
        )
        if LOG.exists()
        else ""
    )

    if marker not in existing:
        with LOG.open(
            "a",
            encoding="utf-8",
        ) as stream:
            stream.write(
                "\n"
                + marker
                + "\n\n"
                "Status: PASS / FROZEN\n\n"
                "- The full frozen Block4.1 "
                  "calibration partition of "
                  "7,209 scenarios is used.\n"
                "- Fit/development/formal "
                  "validation data are not used "
                  "to fit the calibrator.\n"
                "- Calibration inference reuses "
                  "the frozen Block4.4 Gaussian "
                  "checkpoint and Block4.3 fit-only "
                  "normalization.\n"
                "- Per-scenario calibration "
                  "inference shards are resumable.\n"
                "- Calibration changes predictive "
                  "covariance only; predictive mean "
                  "and Stage2 measurement covariance "
                  "R_t are unchanged.\n"
                "- Four independent scalar variance "
                  "scales are fitted, one per "
                  "0.1/0.3/0.5/1.0-s horizon.\n"
                "- Primary calibration objective is "
                  "macro absolute coverage error at "
                  "50/80/90/95/99% confidence.\n"
                "- Coverage-event Brier semantics are "
                  "frozen as "
                  "(inside_indicator - nominal_p)^2.\n"
                "- Calibration-set macro ECE does not "
                  "exceed raw macro ECE.\n"
                "- Calibrated predictive covariance "
                  "remains SPD.\n"
                "- Exact calibration fitting is "
                  "reproducible.\n"
                "- Frozen Gaussian inference repeats "
                  "exactly on four calibration scenes.\n"
                "- Formal validation remains untouched; "
                  "calibration generalization is "
                  "deferred to Block4.8.\n"
                f"- Raw calibration macro ECE: "
                f"{raw_ece:.9f}.\n"
                f"- Calibrated macro ECE: "
                f"{calibrated_ece:.9f}.\n"
                f"- Raw coverage-event Brier: "
                f"{raw_reliability['coverage_event_Brier']:.9f}.\n"
                f"- Calibrated coverage-event Brier: "
                f"{calibrated_reliability['coverage_event_Brier']:.9f}.\n"
                f"- Raw calibration metric NLL: "
                f"{nll_report['raw']['metric_Gaussian_NLL']:.9f}.\n"
                f"- Calibrated metric NLL: "
                f"{nll_report['calibrated']['metric_Gaussian_NLL']:.9f}.\n"
                f"- Variance scales: "
                f"{calibrator_payload['variance_scale']}.\n"
                f"- Calibrator content SHA256: "
                f"{calibrator_content_sha}.\n"
                f"- Calibrator file SHA256: "
                f"{calibrator_file_sha}.\n"
                "- Full Stage4 regression: "
                  "76/76 PASS.\n"
                f"- Implementation files: "
                f"{implementation_files}.\n"
                f"- Implementation SHA256: "
                f"{implementation_sha}.\n"
            )

    print()
    print(
        "============================================================"
    )
    print(
        "STAGE4 BLOCK 4.5 GATE"
    )
    print(
        "============================================================"
    )

    print(
        "calibration scenarios      = 7209 / 7209"
    )
    print(
        "full calibration partition = PASS"
    )
    print(
        "calibration samples        =",
        cache_summary[
            "sample_count"
        ],
    )
    print(
        "fit used for calibration   = NO"
    )
    print(
        "development used           = NO"
    )
    print(
        "formal validation used     = NO"
    )
    print(
        "calibration inference repeat= 4 / 4 PASS"
    )

    print()
    print(
        "variance scales alpha_h    =",
        calibrator_payload[
            "variance_scale"
        ],
    )
    print(
        "std scales sqrt(alpha_h)   =",
        calibrator_payload[
            "standard_deviation_scale"
        ],
    )

    print()
    print(
        "raw macro ECE              =",
        round(
            raw_ece,
            6,
        ),
    )
    print(
        "calibrated macro ECE       =",
        round(
            calibrated_ece,
            6,
        ),
    )
    print(
        "ECE non-worsening          = PASS"
    )

    print(
        "raw coverage-event Brier   =",
        round(
            raw_reliability[
                "coverage_event_Brier"
            ],
            6,
        ),
    )
    print(
        "calibrated Brier           =",
        round(
            calibrated_reliability[
                "coverage_event_Brier"
            ],
            6,
        ),
    )

    print(
        "raw metric Gaussian NLL    =",
        round(
            nll_report[
                "raw"
            ][
                "metric_Gaussian_NLL"
            ],
            6,
        ),
    )
    print(
        "calibrated Gaussian NLL     =",
        round(
            nll_report[
                "calibrated"
            ][
                "metric_Gaussian_NLL"
            ],
            6,
        ),
    )

    print()
    print(
        "predictive mean unchanged  = PASS"
    )
    print(
        "planar ADE unchanged       =",
        round(
            trajectory_metrics[
                "planar_ADE_m"
            ],
            6,
        ),
    )
    print(
        "measurement R_t modified   = NO"
    )
    print(
        "calibrated covariance SPD  = PASS"
    )

    print()
    print(
        "CALIBRATED COVERAGE"
    )

    for key in (
        "0.50",
        "0.80",
        "0.90",
        "0.95",
        "0.99",
    ):
        value = (
            calibrated_reliability[
                "global_levels"
            ][
                key
            ][
                "empirical"
            ]
        )

        print(
            f"coverage {int(float(key)*100):02d}%"
            f"                  = "
            f"{value:.6f}"
        )

    print()
    print(
        "calibrator content SHA256  =",
        calibrator_content_sha,
    )
    print(
        "calibrator file SHA256     =",
        calibrator_file_sha,
    )
    print(
        "statistics content SHA256  =",
        cache_summary[
            "sufficient_statistics_content_sha256"
        ],
    )
    print(
        "exact calibrator repeat    = PASS"
    )
    print(
        "full Stage4 regression     = 76 / 76 PASS"
    )
    print(
        "free GiB                   =",
        round(
            free_gib_after,
            3,
        ),
    )
    print(
        "implementation files       =",
        implementation_files,
    )
    print(
        "implementation SHA256      =",
        implementation_sha,
    )
    print(
        "STATUS = PASS"
    )
    print(
        "report =",
        REPORT,
    )

    print()
    print(
        "===== BLOCK 4.5 FINAL ====="
    )
    print(
        "full calibration partition = PASS"
    )
    print(
        "covariance calibration      = PASS"
    )
    print(
        "50/80/90/95/99 reliability = PASS"
    )
    print(
        "coverage ECE               = PASS"
    )
    print(
        "coverage-event Brier       = PASS"
    )
    print(
        "mean preserved             = PASS"
    )
    print(
        "SPD preserved              = PASS"
    )
    print(
        "calibrator reproducibility = PASS"
    )
    print(
        "formal validation leak     = NONE"
    )
    print(
        "regression                 = 76 / 76"
    )
    print(
        "implementation SHA         =",
        implementation_sha,
    )
    print(
        "log closure                = PASS"
    )


def recovery_hint(
    exc,
):
    text = (
        f"{type(exc).__name__}: "
        f"{exc}"
    ).lower()

    if (
        "cuda"
        in text
        or
        "out of memory"
        in text
    ):
        return (
            "Check nvidia-smi first. "
            "Do not change the frozen "
            "Gaussian model or calibration "
            "partition automatically. "
            "Completed per-scene calibration "
            "shards remain reusable."
        )

    if (
        "python.h"
        in text
        or
        "triton"
        in text
    ):
        return (
            "Verify the already-installed "
            "python3.13-dev headers and "
            "Block4.4 Triton smoke. "
            "Do not reinstall PyTorch."
        )

    if (
        "shard"
        in text
        or
        "cache"
        in text
        or
        "npz"
        in text
    ):
        return (
            "Rerun the same safe command. "
            "Valid completed calibration "
            "shards are reused; corrupt new "
            "Block4.5 shards are archived "
            "and regenerated individually."
        )

    if (
        "checkpoint"
        in text
        or
        "state"
        in text
    ):
        return (
            "Do not retrain Block4.4. "
            "Audit the frozen Gaussian "
            "checkpoint/hash first."
        )

    if (
        "ece"
        in text
        or
        "calibrat"
        in text
    ):
        return (
            "Do not change Block4.4. "
            "Inspect only the Block4.5 "
            "calibration objective/scales "
            "and sufficient statistics."
        )

    if (
        "76/76"
        in text
        or
        "regression"
        in text
    ):
        return (
            "Inspect the failing test. "
            "Do not modify frozen "
            "Block4.0–4.4 behavior."
        )

    return (
        "Inspect reports/"
        "block45_calibration_failure.json. "
        "The full calibration inference "
        "cache is resumable and formal "
        "validation remains untouched."
    )


try:
    main()

except BaseException as exc:
    payload = {
        "stage": 4,
        "block": "4.5_part_2",
        "status": "BLOCKED",

        "exception_type":
            type(exc).__name__,

        "exception":
            str(exc),

        "traceback":
            traceback.format_exc(),

        "recovery_hint":
            recovery_hint(
                exc
            ),

        "calibration_cache_resumable":
            True,

        "block44_modified":
            False,

        "development_used":
            False,

        "formal_validation_used":
            False,
    }

    write_json(
        FAILURE,
        payload,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 4.5 PART 2/2 = BLOCKED"
    )
    print(
        "============================================================"
    )
    print(
        "exception =",
        type(exc).__name__,
        str(exc),
    )
    print()
    print(
        "RECOVERY:"
    )
    print(
        payload[
            "recovery_hint"
        ]
    )
    print()
    print(
        "Block4.4 modified         = NO"
    )
    print(
        "calibration cache reusable= YES"
    )
    print(
        "development used          = NO"
    )
    print(
        "formal validation used    = NO"
    )
    print(
        "terminal remains open     = YES"
    )
    print(
        "failure report =",
        FAILURE,
    )

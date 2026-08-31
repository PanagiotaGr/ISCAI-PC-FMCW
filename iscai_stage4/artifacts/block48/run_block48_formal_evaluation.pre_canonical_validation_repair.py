from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from iscai_stage3.validation import (
    read_motion_scenario,
    read_validation_manifest,
)

from iscai_stage4.data import (
    attach_supervision,
)

from iscai_stage4.data.real_pipeline import (
    build_real_causal_inputs,
    load_frozen_stage2_configs,
)

from iscai_stage4.ml.calibration import (
    apply_variance_scale,
    reliability_metrics,
)

from iscai_stage4.ml.calibration_runtime import (
    CalibrationNormalizer,
    calibrated_nll_per_horizon,
    masked_nll_metrics,
    samples_to_arrays,
)

from iscai_stage4.ml.formal_runtime import (
    construct_deterministic_model,
    eligible_formal_targets,
    extract_deterministic_prediction,
    gmm_formal_metrics,
    resolve_sample_truth_track_index,
    tensor_bundle_sha256,
    trajectory_point_metrics,
)

from iscai_stage4.ml.gaussian_gru import (
    GaussianTrajectoryGRU,
)

from iscai_stage4.ml.gaussian_math import (
    denormalize_gaussian,
    gaussian_nll_per_horizon,
    mahalanobis_squared,
)

from iscai_stage4.ml.gmm_gru import (
    GMMTrajectoryGRU,
)

from iscai_stage4.ml.gmm_math import (
    gmm_joint_nll_per_sample,
)

from iscai_stage4.ml.gmm_runtime import (
    denormalize_gmm,
)

from iscai_stage4.ml.training_utils import (
    CLASS_ID_TO_NAME,
    set_global_determinism,
)


ROOT = Path(
    "/home/agni/waymo"
)

STAGE3 = (
    ROOT
    / "iscai_stage3"
)

STAGE4 = (
    ROOT
    / "iscai_stage4"
)

DATA_PREP = (
    ROOT
    / "iscai_data_prep"
)

PAIRED_ROOT = (
    ROOT
    / "data/"
      "paired_womd_lidar_v1_3_0"
)

PART1 = (
    STAGE4
    / "reports/block48_part1_discovery.json"
)

BLOCK47 = (
    STAGE4
    / "reports/block47_ablation_analysis.json"
)

FORMAL_MANIFEST = (
    STAGE3
    / "artifacts/block38e/"
      "formal_validation_120.jsonl"
)

STAGE3_FORMAL_REPORT = (
    STAGE3
    / "reports/"
      "block38f_formal_evaluation.json"
)

DET_CHECKPOINT = (
    STAGE4
    / "artifacts/block43/"
      "deterministic_gru.pt"
)

GAUSSIAN_CHECKPOINT = (
    STAGE4
    / "artifacts/block44/"
      "gaussian_gru.pt"
)

CALIBRATOR = (
    STAGE4
    / "artifacts/block45/"
      "covariance_scaler.json"
)

GMM_CHECKPOINT = (
    STAGE4
    / "artifacts/block46/"
      "gmm_gru.pt"
)

NORMALIZATION = (
    STAGE4
    / "artifacts/block43/"
      "fit_normalization.json"
)

CACHE_DIR = (
    STAGE4
    / "artifacts/block48/cache"
)

CACHE_REPORT = (
    STAGE4
    / "artifacts/block48/"
      "formal_cache_manifest.json"
)

MERGED = (
    STAGE4
    / "artifacts/block48/"
      "formal_neural_outputs.npz"
)

REPORT = (
    STAGE4
    / "reports/"
      "block48_formal_evaluation.json"
)

FAILURE = (
    STAGE4
    / "reports/"
      "block48_formal_evaluation_failure.json"
)

LOG = (
    STAGE4
    / "docs/"
      "implementation_log.md"
)

EXPECTED_FORMAL_MANIFEST_SHA = (
    "2208e7287ddf6439fda4597c435a9cba"
    "1d1b9d0e4c4547bc5dd92e56e8124e46"
)

EXPECTED_STAGE3_FORMAL_REPORT_SHA = (
    "35ef1353c30f2c871fd79250192ee47b"
    "ed1f8c17962a979336e97585dec80367"
)

EXPECTED_STAGE3_FORMAL_RUN_SHA = (
    "5fbb4642f87ab8e62cf1e1b3ef4214d9"
    "2df5b5a1423bd9fb72b9733ced02e433"
)

EXPECTED_BLOCK47_IMPL_SHA = (
    "4b32edff0c08f23de4ec5109b46b64a8"
    "eccc5357da0deb4fc516b72fe3cad006"
)

EXPECTED_DET_SHA = (
    "5456a76b84d558e9983a59b9f1d3060b"
    "a245e0d36d60883519654f809996dbc5"
)

EXPECTED_GAUSSIAN_SHA = (
    "49ff64d145eaa633f295c16f660df380"
    "c35383e7e3b61279a5aad7cd700d619f"
)

EXPECTED_CALIBRATOR_SHA = (
    "508ff2e3fbcfafe8e001155340c25baaf"
    "3772fe2561a8022a9ed1cf780e66087"
)

EXPECTED_GMM_SHA = (
    "5aebeaf40d522d58345d424ae2557d6f"
    "87e2c02bdc26c23635cc6e3a28cbe3ee"
)

MIN_FREE_GIB = 250.0


def file_sha256(
    path,
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


def write_json(
    path,
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
    path,
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


def implementation_fingerprint():
    roots = (
        STAGE4 / "src",
        STAGE4 / "tests",
        STAGE4 / "configs",
        STAGE4 / "scripts",
    )

    paths = []

    for root in roots:
        for path in root.rglob(
            "*"
        ):
            if not path.is_file():
                continue

            if "__pycache__" in path.parts:
                continue

            if path.suffix in (
                ".pyc",
                ".pyo",
            ):
                continue

            paths.append(
                path
            )

    paths.sort(
        key=lambda path:
            str(
                path.relative_to(
                    STAGE4
                )
            )
    )

    digest = sha256()

    for path in paths:
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
        len(
            paths
        ),
        digest.hexdigest(),
    )


def read_formal_rows():
    rows = tuple(
        json.loads(
            line
        )
        for line in (
            FORMAL_MANIFEST
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        )
        if line.strip()
    )

    if len(rows) != 120:
        raise RuntimeError(
            "Formal manifest is not N=120."
        )

    return rows


def json_paths(
    value,
):
    found = []

    if isinstance(
        value,
        dict,
    ):
        for item in value.values():
            found.extend(
                json_paths(
                    item
                )
            )

    elif isinstance(
        value,
        list,
    ):
        for item in value:
            found.extend(
                json_paths(
                    item
                )
            )

    elif isinstance(
        value,
        str,
    ):
        if (
            "validation"
            in value.lower()
            and
            (
                value.endswith(
                    ".jsonl"
                )
                or
                value.endswith(
                    ".json"
                )
            )
        ):
            found.append(
                Path(
                    value
                )
            )

    return found


def validation_manifest_candidates():
    candidates = []

    for json_file in (
        STAGE3
        / "reports"
    ).glob(
        "*.json"
    ):
        try:
            payload = json.loads(
                json_file.read_text(
                    encoding="utf-8"
                )
            )

        except Exception:
            continue

        candidates.extend(
            json_paths(
                payload
            )
        )

    patterns = (
        "*validation*manifest*.jsonl",
        "*validation*manifest*.json",
    )

    roots = (
        STAGE3,
        DATA_PREP,
        PAIRED_ROOT,
    )

    for root in roots:
        if not root.exists():
            continue

        for pattern in patterns:
            try:
                candidates.extend(
                    root.rglob(
                        pattern
                    )
                )

            except Exception:
                pass

    unique = []

    seen = set()

    for path in candidates:
        if not path.is_absolute():
            path = (
                ROOT
                /
                path
            )

        try:
            resolved = path.resolve()

        except Exception:
            continue

        if (
            resolved in seen
            or
            not resolved.is_file()
        ):
            continue

        seen.add(
            resolved
        )

        unique.append(
            resolved
        )

    return tuple(
        unique
    )


def resolve_validation_manifest(
    formal_rows,
):
    formal_ids = {
        row[
            "scenario_id"
        ]
        for row in formal_rows
    }

    matches = []

    diagnostics = []

    for path in (
        validation_manifest_candidates()
    ):
        try:
            rows = read_validation_manifest(
                path
            )

            by_id = {
                row.scenario_id:
                    row
                for row in rows
            }

            overlap = len(
                formal_ids
                &
                set(
                    by_id
                )
            )

            diagnostics.append(
                (
                    str(
                        path
                    ),
                    len(
                        rows
                    ),
                    overlap,
                )
            )

            if formal_ids.issubset(
                by_id
            ):
                matches.append(
                    (
                        path,
                        rows,
                        by_id,
                    )
                )

        except Exception:
            continue

    if not matches:
        raise RuntimeError(
            "No official validation manifest "
            "contains all frozen formal IDs. "
            f"Candidates={diagnostics[:30]}"
        )

    content_groups = {}

    for path, rows, by_id in (
        matches
    ):
        digest = file_sha256(
            path
        )

        content_groups.setdefault(
            digest,
            []
        ).append(
            (
                path,
                rows,
                by_id,
            )
        )

    if len(
        content_groups
    ) != 1:
        evidence = {
            digest:
                [
                    str(
                        item[
                            0
                        ]
                    )
                    for item in group
                ]
            for digest, group
            in content_groups.items()
        }

        raise RuntimeError(
            "Multiple non-identical validation "
            "manifests contain all formal IDs: "
            +
            repr(
                evidence
            )
        )

    group = next(
        iter(
            content_groups.values()
        )
    )

    group = sorted(
        group,
        key=lambda item:
            len(
                str(
                    item[
                        0
                    ]
                )
            ),
    )

    return group[
        0
    ]


def build_gaussian(
    architecture,
    *,
    device,
):
    section = architecture[
        "model"
    ]

    return GaussianTrajectoryGRU(
        target_hidden_dim=int(
            section[
                "target_hidden_dim"
            ]
        ),
        neighbor_hidden_dim=int(
            section[
                "neighbor_hidden_dim"
            ]
        ),
        map_hidden_dim=int(
            section[
                "map_hidden_dim"
            ]
        ),
        fusion_hidden_dim=int(
            section[
                "fusion_hidden_dim"
            ]
        ),
        use_neighbors=True,
        use_map=True,
    ).to(
        device
    )


def build_gmm(
    architecture,
    *,
    device,
):
    section = architecture[
        "model"
    ]

    return GMMTrajectoryGRU(
        target_hidden_dim=int(
            section[
                "target_hidden_dim"
            ]
        ),
        neighbor_hidden_dim=int(
            section[
                "neighbor_hidden_dim"
            ]
        ),
        map_hidden_dim=int(
            section[
                "map_hidden_dim"
            ]
        ),
        fusion_hidden_dim=int(
            section[
                "fusion_hidden_dim"
            ]
        ),
        use_neighbors=True,
        use_map=True,
    ).to(
        device
    )


def cache_path(
    rank,
    scenario_id,
):
    return (
        CACHE_DIR
        /
        (
            f"{rank:03d}_"
            f"{scenario_id}.npz"
        )
    )


def cache_contract_sha():
    payload = {
        "formal_manifest":
            EXPECTED_FORMAL_MANIFEST_SHA,

        "deterministic":
            EXPECTED_DET_SHA,

        "Gaussian":
            EXPECTED_GAUSSIAN_SHA,

        "calibrator":
            EXPECTED_CALIBRATOR_SHA,

        "GMM":
            EXPECTED_GMM_SHA,

        "Block47":
            EXPECTED_BLOCK47_IMPL_SHA,

        "Stage3_formal":
            EXPECTED_STAGE3_FORMAL_RUN_SHA,
    }

    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        ).encode(
            "utf-8"
        )
    ).hexdigest()


CACHE_CONTRACT = (
    cache_contract_sha()
)


def validate_shard(
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
            if (
                str(
                    data[
                        "scenario_id"
                    ].item()
                )
                !=
                scenario_id
            ):
                return False

            if (
                str(
                    data[
                        "cache_contract_sha256"
                    ].item()
                )
                !=
                CACHE_CONTRACT
            ):
                return False

            truth = np.asarray(
                data[
                    "truth"
                ]
            )

            validity = np.asarray(
                data[
                    "validity"
                ]
            )

            selected_truth_index = np.asarray(
                data[
                    "selected_truth_index"
                ]
            )

            eligible_index = np.asarray(
                data[
                    "eligible_truth_index"
                ]
            )

            if truth.shape != (
                len(
                    selected_truth_index
                ),
                4,
                3,
            ):
                return False

            if validity.shape != (
                len(
                    selected_truth_index
                ),
                4,
            ):
                return False

            if np.asarray(
                data[
                    "det_mean"
                ]
            ).shape != truth.shape:
                return False

            if np.asarray(
                data[
                    "gaussian_mean"
                ]
            ).shape != truth.shape:
                return False

            if np.asarray(
                data[
                    "gaussian_scale"
                ]
            ).shape != (
                len(
                    selected_truth_index
                ),
                4,
                3,
                3,
            ):
                return False

            if np.asarray(
                data[
                    "gmm_logits"
                ]
            ).shape != (
                len(
                    selected_truth_index
                ),
                3,
            ):
                return False

            if np.asarray(
                data[
                    "gmm_means"
                ]
            ).shape != (
                len(
                    selected_truth_index
                ),
                3,
                4,
                3,
            ):
                return False

            if np.asarray(
                data[
                    "eligible_validity"
                ]
            ).shape != (
                len(
                    eligible_index
                ),
                4,
            ):
                return False

            if not bool(
                data[
                    "tracks_to_predict_accessed_after_prediction"
                ].item()
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
            f".corrupt.{int(time.time())}"
        )
    )

    os.replace(
        path,
        destination,
    )


def infer_scene(
    formal_row,
    validation_row,
    *,
    normalizer,
    deterministic_model,
    Gaussian_model,
    GMM_model,
    clean_config,
    degraded_config,
    device,
):
    scenario = read_motion_scenario(
        validation_row,
        paired_root=PAIRED_ROOT,
        compact_record_offset=int(
            formal_row[
                "compact_record_offset"
            ]
        ),
    )

    if (
        scenario.scenario_id
        !=
        formal_row[
            "scenario_id"
        ]
    ):
        raise RuntimeError(
            "Formal scenario ID mismatch."
        )

    built = build_real_causal_inputs(
        scenario,
        clean_config=clean_config,
        degraded_config=degraded_config,
    )

    samples = attach_supervision(
        built[
            "scene_inputs"
        ],
        scenario,
        T_H0_from_W=(
            built[
                "adapted"
            ]
            .frames
            .T_H0_from_W
        ),
    )

    truth_metadata = []

    for sample in samples:
        resolved = (
            resolve_sample_truth_track_index(
                sample
            )
        )

        truth_metadata.append(
            resolved
        )

    truth_indices_all = np.asarray(
        [
            item[
                "index"
            ]
            for item in (
                truth_metadata
            )
        ],
        dtype=np.int64,
    )

    if (
        len(
            np.unique(
                truth_indices_all
            )
        )
        !=
        len(
            truth_indices_all
        )
    ):
        raise RuntimeError(
            "Frozen supervision produced "
            "duplicate truth-track matches."
        )

    if samples:
        arrays = samples_to_arrays(
            samples
        )

        indices = np.arange(
            len(
                samples
            ),
            dtype=np.int64,
        )

        batch = normalizer.prepare(
            arrays,
            device=device,
        )

        deterministic_model.eval()
        Gaussian_model.eval()
        GMM_model.eval()

        with torch.inference_mode():
            det_output = (
                deterministic_model(
                    batch[
                        "target"
                    ],
                    batch[
                        "neighbors"
                    ],
                    batch[
                        "neighbor_mask"
                    ],
                    batch[
                        "map_context"
                    ],
                )
            )

            det_normalized = (
                extract_deterministic_prediction(
                    det_output
                )
            )

            Gaussian_output = (
                Gaussian_model(
                    batch[
                        "target"
                    ],
                    batch[
                        "neighbors"
                    ],
                    batch[
                        "neighbor_mask"
                    ],
                    batch[
                        "map_context"
                    ],
                )
            )

            GMM_output = (
                GMM_model(
                    batch[
                        "target"
                    ],
                    batch[
                        "neighbors"
                    ],
                    batch[
                        "neighbor_mask"
                    ],
                    batch[
                        "map_context"
                    ],
                )
            )

            # Prediction hash is computed BEFORE
            # tracks_to_predict is accessed.
            all_prediction_sha = (
                tensor_bundle_sha256(
                    (
                        (
                            "det",
                            det_normalized,
                        ),
                        (
                            "gaussian_mean",
                            Gaussian_output.mean,
                        ),
                        (
                            "gaussian_scale",
                            Gaussian_output.scale_tril,
                        ),
                        (
                            "gmm_logits",
                            GMM_output.mixture_logits,
                        ),
                        (
                            "gmm_means",
                            GMM_output.means,
                        ),
                        (
                            "gmm_scale",
                            GMM_output.scale_tril,
                        ),
                    )
                )
            )

            det_metric = (
                det_normalized
                *
                normalizer.label_std[
                    None,
                    :,
                    :
                ]
                +
                normalizer.label_mean[
                    None,
                    :,
                    :
                ]
            )

            (
                Gaussian_mean_metric,
                Gaussian_scale_metric,
            ) = denormalize_gaussian(
                Gaussian_output.mean,
                Gaussian_output.scale_tril,
                normalizer.label_mean,
                normalizer.label_std,
            )

            (
                GMM_means_metric,
                GMM_scale_metric,
            ) = denormalize_gmm(
                GMM_output.means,
                GMM_output.scale_tril,
                normalizer.label_mean,
                normalizer.label_std,
            )

        truth_all = (
            batch[
                "true_displacement"
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        validity_all = (
            batch[
                "future_mask"
            ]
            .detach()
            .cpu()
            .numpy()
            >
            0.5
        )

        class_all = (
            batch[
                "class_id"
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.int64,
                copy=False,
            )
        )

        det_metric = (
            det_metric.detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        Gaussian_mean_metric = (
            Gaussian_mean_metric.detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        Gaussian_scale_metric = (
            Gaussian_scale_metric.detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        GMM_logits = (
            GMM_output.mixture_logits
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        GMM_means_metric = (
            GMM_means_metric.detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        GMM_scale_metric = (
            GMM_scale_metric.detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

    else:
        all_prediction_sha = (
            sha256(
                b""
            ).hexdigest()
        )

        truth_all = np.zeros(
            (
                0,
                4,
                3,
            ),
            dtype=np.float32,
        )

        validity_all = np.zeros(
            (
                0,
                4,
            ),
            dtype=np.bool_,
        )

        class_all = np.zeros(
            (
                0,
            ),
            dtype=np.int64,
        )

        det_metric = np.zeros_like(
            truth_all
        )

        Gaussian_mean_metric = (
            np.zeros_like(
                truth_all
            )
        )

        Gaussian_scale_metric = (
            np.zeros(
                (
                    0,
                    4,
                    3,
                    3,
                ),
                dtype=np.float32,
            )
        )

        GMM_logits = np.zeros(
            (
                0,
                3,
            ),
            dtype=np.float32,
        )

        GMM_means_metric = np.zeros(
            (
                0,
                3,
                4,
                3,
            ),
            dtype=np.float32,
        )

        GMM_scale_metric = np.zeros(
            (
                0,
                3,
                4,
                3,
                3,
            ),
            dtype=np.float32,
        )

    # ========================================================
    # Evaluator target metadata accessed only AFTER all
    # neural predictions above have already been generated.
    # ========================================================

    eligible = eligible_formal_targets(
        scenario
    )

    target_set = set(
        eligible[
            "track_index"
        ].tolist()
    )

    selected_mask = np.asarray(
        [
            int(
                index
            )
            in
            target_set
            for index in (
                truth_indices_all
            )
        ],
        dtype=np.bool_,
    )

    selected_truth_index = (
        truth_indices_all[
            selected_mask
        ]
    )

    if (
        len(
            np.unique(
                selected_truth_index
            )
        )
        !=
        len(
            selected_truth_index
        )
    ):
        raise RuntimeError(
            "Formal neural matches are "
            "not one-to-one."
        )

    # Cross-check class identity against WOMD truth track.
    for local_index, truth_index in enumerate(
        selected_truth_index.tolist()
    ):
        expected_class = int(
            scenario.tracks[
                truth_index
            ].object_type
        )

        actual_class = int(
            class_all[
                selected_mask
            ][
                local_index
            ]
        )

        if (
            actual_class
            !=
            expected_class
        ):
            raise RuntimeError(
                "Neural sample actor class does "
                "not match WOMD truth track."
            )

    return {
        "scenario_id":
            scenario.scenario_id,

        "sample_schema":
            {
                "sample_type":
                    (
                        type(
                            samples[
                                0
                            ]
                        ).__name__
                        if samples
                        else None
                    ),

                "sample_public_fields":
                    (
                        list(
                            public_name
                            for public_name
                            in
                            []
                        )
                    ),
            },

        "truth_index_source":
            (
                truth_metadata[
                    0
                ][
                    "source"
                ]
                if truth_metadata
                else None
            ),

        "all_prediction_sha256":
            all_prediction_sha,

        "selected_truth_index":
            selected_truth_index,

        "selected_class_id":
            class_all[
                selected_mask
            ],

        "truth":
            truth_all[
                selected_mask
            ],

        "validity":
            validity_all[
                selected_mask
            ],

        "det_mean":
            det_metric[
                selected_mask
            ],

        "gaussian_mean":
            Gaussian_mean_metric[
                selected_mask
            ],

        "gaussian_scale":
            Gaussian_scale_metric[
                selected_mask
            ],

        "gmm_logits":
            GMM_logits[
                selected_mask
            ],

        "gmm_means":
            GMM_means_metric[
                selected_mask
            ],

        "gmm_scale":
            GMM_scale_metric[
                selected_mask
            ],

        "eligible_truth_index":
            eligible[
                "track_index"
            ],

        "eligible_class_id":
            eligible[
                "class_id"
            ],

        "eligible_validity":
            eligible[
                "validity"
            ],

        "all_supervised_prediction_count":
            int(
                len(
                    samples
                )
            ),

        "matched_formal_target_count":
            int(
                selected_mask.sum()
            ),

        "eligible_formal_target_count":
            int(
                len(
                    eligible[
                        "track_index"
                    ]
                )
            ),

        "tracks_to_predict_accessed_after_prediction":
            True,
    }


def save_scene_shard(
    rank,
    payload,
):
    path = cache_path(
        rank,
        payload[
            "scenario_id"
        ],
    )

    atomic_npz(
        path,

        scenario_id=np.asarray(
            payload[
                "scenario_id"
            ]
        ),

        cache_contract_sha256=np.asarray(
            CACHE_CONTRACT
        ),

        all_prediction_sha256=np.asarray(
            payload[
                "all_prediction_sha256"
            ]
        ),

        truth_index_source=np.asarray(
            payload[
                "truth_index_source"
            ]
            or
            ""
        ),

        selected_truth_index=(
            payload[
                "selected_truth_index"
            ]
        ),

        selected_class_id=(
            payload[
                "selected_class_id"
            ]
        ),

        truth=(
            payload[
                "truth"
            ]
        ),

        validity=(
            payload[
                "validity"
            ]
        ),

        det_mean=(
            payload[
                "det_mean"
            ]
        ),

        gaussian_mean=(
            payload[
                "gaussian_mean"
            ]
        ),

        gaussian_scale=(
            payload[
                "gaussian_scale"
            ]
        ),

        gmm_logits=(
            payload[
                "gmm_logits"
            ]
        ),

        gmm_means=(
            payload[
                "gmm_means"
            ]
        ),

        gmm_scale=(
            payload[
                "gmm_scale"
            ]
        ),

        eligible_truth_index=(
            payload[
                "eligible_truth_index"
            ]
        ),

        eligible_class_id=(
            payload[
                "eligible_class_id"
            ]
        ),

        eligible_validity=(
            payload[
                "eligible_validity"
            ]
        ),

        all_supervised_prediction_count=np.asarray(
            payload[
                "all_supervised_prediction_count"
            ],
            dtype=np.int64,
        ),

        matched_formal_target_count=np.asarray(
            payload[
                "matched_formal_target_count"
            ],
            dtype=np.int64,
        ),

        eligible_formal_target_count=np.asarray(
            payload[
                "eligible_formal_target_count"
            ],
            dtype=np.int64,
        ),

        tracks_to_predict_accessed_after_prediction=np.asarray(
            True,
            dtype=np.bool_,
        ),
    )

    if not validate_shard(
        path,
        payload[
            "scenario_id"
        ],
    ):
        raise RuntimeError(
            "New formal neural shard "
            "failed validation."
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
                "selected_truth_index",
                "selected_class_id",
                "truth",
                "validity",
                "det_mean",
                "gaussian_mean",
                "gaussian_scale",
                "gmm_logits",
                "gmm_means",
                "gmm_scale",
                "eligible_truth_index",
                "eligible_class_id",
                "eligible_validity",
            )
        } | {
            "all_prediction_sha256":
                str(
                    data[
                        "all_prediction_sha256"
                    ].item()
                ),

            "truth_index_source":
                str(
                    data[
                        "truth_index_source"
                    ].item()
                ),

            "all_supervised_prediction_count":
                int(
                    data[
                        "all_supervised_prediction_count"
                    ].item()
                ),

            "matched_formal_target_count":
                int(
                    data[
                        "matched_formal_target_count"
                    ].item()
                ),

            "eligible_formal_target_count":
                int(
                    data[
                        "eligible_formal_target_count"
                    ].item()
                ),
        }


def merge_shards(
    formal_rows,
):
    keys = (
        "selected_class_id",
        "truth",
        "validity",
        "det_mean",
        "gaussian_mean",
        "gaussian_scale",
        "gmm_logits",
        "gmm_means",
        "gmm_scale",
        "eligible_class_id",
        "eligible_validity",
    )

    parts = {
        key: []
        for key in keys
    }

    scenario_namespace_matched = []
    scenario_namespace_eligible = []

    total_supervised = 0

    prediction_hashes = []

    truth_index_sources = set()

    for rank, row in enumerate(
        formal_rows,
        start=1,
    ):
        path = cache_path(
            rank,
            row[
                "scenario_id"
            ],
        )

        if not validate_shard(
            path,
            row[
                "scenario_id"
            ],
        ):
            raise RuntimeError(
                f"Invalid formal shard: {path}"
            )

        shard = load_shard(
            path
        )

        for key in keys:
            parts[
                key
            ].append(
                shard[
                    key
                ]
            )

        scenario_namespace_matched.extend(
            (
                rank,
                int(
                    index
                ),
            )
            for index in (
                shard[
                    "selected_truth_index"
                ].tolist()
            )
        )

        scenario_namespace_eligible.extend(
            (
                rank,
                int(
                    index
                ),
            )
            for index in (
                shard[
                    "eligible_truth_index"
                ].tolist()
            )
        )

        total_supervised += int(
            shard[
                "all_supervised_prediction_count"
            ]
        )

        prediction_hashes.append(
            shard[
                "all_prediction_sha256"
            ]
        )

        if shard[
            "truth_index_source"
        ]:
            truth_index_sources.add(
                shard[
                    "truth_index_source"
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
        for key in keys
    }

    atomic_npz(
        MERGED,
        **merged,
    )

    digest = sha256()

    for key in keys:
        array = np.ascontiguousarray(
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
            array.tobytes(
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

    return (
        merged,
        {
            "formal_scenarios":
                120,

            "all_supervised_predictions":
                total_supervised,

            "matched_formal_targets":
                len(
                    scenario_namespace_matched
                ),

            "eligible_formal_targets":
                len(
                    scenario_namespace_eligible
                ),

            "matched_namespace":
                scenario_namespace_matched,

            "eligible_namespace":
                scenario_namespace_eligible,

            "truth_index_sources":
                sorted(
                    truth_index_sources
                ),

            "merged_file_sha256":
                file_sha256(
                    MERGED
                ),

            "merged_content_sha256":
                digest.hexdigest(),
        },
    )


def classical_metrics(
    Stage3_payload,
):
    metrics = (
        Stage3_payload[
            "primary_evaluation"
        ][
            "metrics"
        ]
    )

    result = {}

    for name, payload in (
        metrics.items()
    ):
        if (
            not isinstance(
                payload,
                dict,
            )
            or
            "ade_m"
            not in payload
        ):
            continue

        result[
            name
        ] = {
            "ade_m":
                float(
                    payload[
                        "ade_m"
                    ]
                ),

            "fde_m":
                (
                    float(
                        payload[
                            "fde_m"
                        ]
                    )
                    if payload.get(
                        "fde_m"
                    )
                    is not None
                    else None
                ),

            "recall":
                (
                    float(
                        payload[
                            "reconstruction_recall"
                        ]
                    )
                    if payload.get(
                        "reconstruction_recall"
                    )
                    is not None
                    else None
                ),

            "f1":
                (
                    float(
                        payload[
                            "reconstruction_f1"
                        ]
                    )
                    if payload.get(
                        "reconstruction_f1"
                    )
                    is not None
                    else None
                ),
        }

    return result


def main():
    print(
        "============================================================"
    )
    print(
        "STAGE 4 — BLOCK 4.8 "
        "FROZEN FORMAL N=120 EVALUATION"
    )
    print(
        "============================================================"
    )

    required = (
        PART1,
        BLOCK47,
        FORMAL_MANIFEST,
        STAGE3_FORMAL_REPORT,
        DET_CHECKPOINT,
        GAUSSIAN_CHECKPOINT,
        CALIBRATOR,
        GMM_CHECKPOINT,
        NORMALIZATION,
    )

    missing = [
        str(
            path
        )
        for path in required
        if not path.is_file()
    ]

    if missing:
        raise RuntimeError(
            "Missing frozen artifact(s): "
            +
            ", ".join(
                missing
            )
        )

    part1 = json.loads(
        PART1.read_text(
            encoding="utf-8"
        )
    )

    block47 = json.loads(
        BLOCK47.read_text(
            encoding="utf-8"
        )
    )

    Stage3_formal = json.loads(
        STAGE3_FORMAL_REPORT.read_text(
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
            "Block4.8 Part1 is not PASS."
        )

    if (
        block47.get(
            "status"
        )
        !=
        "PASS"
    ):
        raise RuntimeError(
            "Block4.7 is not PASS."
        )

    if (
        block47[
            "implementation"
        ][
            "sha256"
        ]
        !=
        EXPECTED_BLOCK47_IMPL_SHA
    ):
        raise RuntimeError(
            "Frozen Block4.7 implementation "
            "SHA changed."
        )

    checks = (
        (
            FORMAL_MANIFEST,
            EXPECTED_FORMAL_MANIFEST_SHA,
            "formal manifest",
        ),
        (
            STAGE3_FORMAL_REPORT,
            EXPECTED_STAGE3_FORMAL_REPORT_SHA,
            "Stage3 formal report",
        ),
        (
            DET_CHECKPOINT,
            EXPECTED_DET_SHA,
            "deterministic checkpoint",
        ),
        (
            GAUSSIAN_CHECKPOINT,
            EXPECTED_GAUSSIAN_SHA,
            "Gaussian checkpoint",
        ),
        (
            CALIBRATOR,
            EXPECTED_CALIBRATOR_SHA,
            "calibrator",
        ),
        (
            GMM_CHECKPOINT,
            EXPECTED_GMM_SHA,
            "GMM checkpoint",
        ),
    )

    for path, expected, name in checks:
        actual = file_sha256(
            path
        )

        if actual != expected:
            raise RuntimeError(
                f"Frozen {name} SHA changed."
            )

    if (
        Stage3_formal[
            "deterministic"
        ][
            "run_sha256"
        ]
        !=
        EXPECTED_STAGE3_FORMAL_RUN_SHA
    ):
        raise RuntimeError(
            "Frozen Stage3 formal run "
            "SHA changed."
        )

    normalization = json.loads(
        NORMALIZATION.read_text(
            encoding="utf-8"
        )
    )

    if (
        normalization.get(
            "source"
        )
        !=
        "fit_only"
    ):
        raise RuntimeError(
            "Normalization is not fit-only."
        )

    if not PAIRED_ROOT.is_dir():
        raise RuntimeError(
            "Paired validation root is missing."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable."
        )

    device = torch.device(
        "cuda:0"
    )

    set_global_determinism(
        20260824
    )

    formal_rows = read_formal_rows()

    (
        validation_manifest_path,
        validation_rows,
        validation_by_id,
    ) = resolve_validation_manifest(
        formal_rows
    )

    print(
        "formal manifest N          = 120 PASS"
    )
    print(
        "formal manifest SHA        = PASS"
    )
    print(
        "validation manifest        =",
        validation_manifest_path,
    )
    print(
        "validation manifest rows   =",
        len(
            validation_rows
        ),
    )
    print(
        "all formal IDs resolvable = PASS"
    )
    print(
        "Stage3 formal run SHA      = PASS"
    )
    print(
        "fit-only normalization     = PASS"
    )
    print(
        "formal training            = NO"
    )
    print(
        "formal model selection     = NO"
    )
    print(
        "formal calibration fit     = NO"
    )

    deterministic_checkpoint = torch.load(
        DET_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    Gaussian_checkpoint = torch.load(
        GAUSSIAN_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    GMM_checkpoint = torch.load(
        GMM_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    architecture = (
        deterministic_checkpoint[
            "configuration"
        ]
    )

    (
        deterministic_model,
        deterministic_class,
    ) = construct_deterministic_model(
        architecture,
        deterministic_checkpoint[
            "state_dict"
        ],
        device=device,
    )

    Gaussian_model = build_gaussian(
        architecture,
        device=device,
    )

    Gaussian_model.load_state_dict(
        Gaussian_checkpoint[
            "state_dict"
        ],
        strict=True,
    )

    GMM_architecture = (
        GMM_checkpoint[
            "architecture_configuration"
        ]
    )

    GMM_model = build_gmm(
        GMM_architecture,
        device=device,
    )

    GMM_model.load_state_dict(
        GMM_checkpoint[
            "state_dict"
        ],
        strict=True,
    )

    print()
    print(
        "deterministic class        =",
        deterministic_class,
    )
    print(
        "deterministic state load   = PASS"
    )
    print(
        "Gaussian state load        = PASS"
    )
    print(
        "GMM state load             = PASS"
    )

    normalizer = (
        CalibrationNormalizer(
            normalization,
            device=device,
        )
    )

    calibrator = json.loads(
        CALIBRATOR.read_text(
            encoding="utf-8"
        )
    )

    variance_scale = tuple(
        float(
            value
        )
        for value in (
            calibrator[
                "variance_scale"
            ]
        )
    )

    (
        clean_config,
        degraded_config,
    ) = load_frozen_stage2_configs()

    # ========================================================
    # Schema probe BEFORE the full 120-scenario scan.
    # ========================================================

    print()
    print(
        "===== FORMAL SAMPLE SCHEMA PROBE ====="
    )

    probe_payload = None
    probe_rank = None

    for rank, formal_row in enumerate(
        formal_rows,
        start=1,
    ):
        probe = infer_scene(
            formal_row,
            validation_by_id[
                formal_row[
                    "scenario_id"
                ]
            ],
            normalizer=normalizer,
            deterministic_model=(
                deterministic_model
            ),
            Gaussian_model=(
                Gaussian_model
            ),
            GMM_model=(
                GMM_model
            ),
            clean_config=clean_config,
            degraded_config=degraded_config,
            device=device,
        )

        if (
            probe[
                "all_supervised_prediction_count"
            ]
            >
            0
        ):
            probe_payload = probe
            probe_rank = rank
            break

    if probe_payload is None:
        raise RuntimeError(
            "No supervised sample found "
            "in frozen N=120 formal set."
        )

    print(
        "probe scenario             =",
        probe_payload[
            "scenario_id"
        ],
    )
    print(
        "truth-index metadata source=",
        probe_payload[
            "truth_index_source"
        ],
    )
    print(
        "all causal predictions     =",
        probe_payload[
            "all_supervised_prediction_count"
        ],
    )
    print(
        "eligible formal targets    =",
        probe_payload[
            "eligible_formal_target_count"
        ],
    )
    print(
        "matched formal targets     =",
        probe_payload[
            "matched_formal_target_count"
        ],
    )
    print(
        "tracks_to_predict access   = AFTER PREDICTION PASS"
    )

    # Save probe immediately so it is not recomputed.
    save_scene_shard(
        probe_rank,
        probe_payload,
    )

    # ========================================================
    # Resumable formal inference scan
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "FORMAL N=120 NEURAL INFERENCE CACHE"
    )
    print(
        "============================================================"
    )

    reused = 0
    matched_total = 0
    eligible_total = 0

    cache_rows = []

    started = time.perf_counter()

    for rank, formal_row in enumerate(
        formal_rows,
        start=1,
    ):
        scenario_id = formal_row[
            "scenario_id"
        ]

        path = cache_path(
            rank,
            scenario_id,
        )

        if validate_shard(
            path,
            scenario_id,
        ):
            reused += 1

            shard = load_shard(
                path
            )

            matched = int(
                shard[
                    "matched_formal_target_count"
                ]
            )

            eligible = int(
                shard[
                    "eligible_formal_target_count"
                ]
            )

            prediction_sha = (
                shard[
                    "all_prediction_sha256"
                ]
            )

        else:
            archive_invalid(
                path
            )

            payload = infer_scene(
                formal_row,
                validation_by_id[
                    scenario_id
                ],
                normalizer=normalizer,
                deterministic_model=(
                    deterministic_model
                ),
                Gaussian_model=(
                    Gaussian_model
                ),
                GMM_model=(
                    GMM_model
                ),
                clean_config=clean_config,
                degraded_config=degraded_config,
                device=device,
            )

            save_scene_shard(
                rank,
                payload,
            )

            matched = int(
                payload[
                    "matched_formal_target_count"
                ]
            )

            eligible = int(
                payload[
                    "eligible_formal_target_count"
                ]
            )

            prediction_sha = (
                payload[
                    "all_prediction_sha256"
                ]
            )

        matched_total += matched
        eligible_total += eligible

        cache_rows.append({
            "formal_rank":
                rank,

            "scenario_id":
                scenario_id,

            "matched_targets":
                matched,

            "eligible_targets":
                eligible,

            "all_prediction_sha256":
                prediction_sha,
        })

        if (
            rank % 10 == 0
            or
            rank == 120
        ):
            print(
                f"formal: {rank}/120 "
                f"| matched={matched_total} "
                f"| eligible={eligible_total} "
                f"| reused={reused}",
                flush=True,
            )

    cache_runtime_s = (
        time.perf_counter()
        -
        started
    )

    (
        merged,
        merge_summary,
    ) = merge_shards(
        formal_rows
    )

    write_json(
        CACHE_REPORT,
        {
            "stage":
                4,

            "block":
                "4.8",

            "formal_scenarios":
                120,

            "resumable":
                True,

            "tracks_to_predict_accessed_after_prediction":
                True,

            "validation_manifest":
                str(
                    validation_manifest_path
                ),

            "validation_manifest_sha256":
                file_sha256(
                    validation_manifest_path
                ),

            "cache_contract_sha256":
                CACHE_CONTRACT,

            "reused_scenarios":
                reused,

            "runtime_s":
                cache_runtime_s,

            "summary":
                merge_summary,

            "scenarios":
                cache_rows,
        },
    )

    print()
    print(
        "matched formal targets     =",
        merge_summary[
            "matched_formal_targets"
        ],
    )
    print(
        "eligible formal targets    =",
        merge_summary[
            "eligible_formal_targets"
        ],
    )
    print(
        "merged formal output SHA   =",
        merge_summary[
            "merged_content_sha256"
        ],
    )

    # ========================================================
    # Exact repeat on first four non-empty scenes
    # ========================================================

    print()
    print(
        "===== EXACT FORMAL INFERENCE REPEAT ====="
    )

    repeated = 0

    for rank, formal_row in enumerate(
        formal_rows,
        start=1,
    ):
        shard = load_shard(
            cache_path(
                rank,
                formal_row[
                    "scenario_id"
                ],
            )
        )

        if (
            shard[
                "all_supervised_prediction_count"
            ]
            <=
            0
        ):
            continue

        repeat = infer_scene(
            formal_row,
            validation_by_id[
                formal_row[
                    "scenario_id"
                ]
            ],
            normalizer=normalizer,
            deterministic_model=(
                deterministic_model
            ),
            Gaussian_model=(
                Gaussian_model
            ),
            GMM_model=(
                GMM_model
            ),
            clean_config=clean_config,
            degraded_config=degraded_config,
            device=device,
        )

        if (
            repeat[
                "all_prediction_sha256"
            ]
            !=
            shard[
                "all_prediction_sha256"
            ]
        ):
            raise RuntimeError(
                "Formal prediction hash "
                "repeat mismatch."
            )

        repeated += 1

        if repeated == 4:
            break

    if repeated != 4:
        raise RuntimeError(
            "Could not obtain four formal "
            "repeat-inference probes."
        )

    print(
        "formal inference repeat    = 4 / 4 PASS"
    )

    # ========================================================
    # Point-prediction metrics
    # ========================================================

    truth = merged[
        "truth"
    ]

    validity = (
        merged[
            "validity"
        ]
        >
        0.5
    )

    classes = merged[
        "selected_class_id"
    ]

    deterministic_metrics = (
        trajectory_point_metrics(
            merged[
                "det_mean"
            ],
            truth,
            validity,
            classes,
            class_id_to_name=(
                CLASS_ID_TO_NAME
            ),
        )
    )

    Gaussian_point_metrics = (
        trajectory_point_metrics(
            merged[
                "gaussian_mean"
            ],
            truth,
            validity,
            classes,
            class_id_to_name=(
                CLASS_ID_TO_NAME
            ),
        )
    )

    # ========================================================
    # Raw + calibrated Gaussian probabilistic metrics
    # ========================================================

    mean_t = torch.from_numpy(
        np.asarray(
            merged[
                "gaussian_mean"
            ],
            dtype=np.float64,
        )
    )

    scale_t = torch.from_numpy(
        np.asarray(
            merged[
                "gaussian_scale"
            ],
            dtype=np.float64,
        )
    )

    truth_t = torch.from_numpy(
        np.asarray(
            truth,
            dtype=np.float64,
        )
    )

    raw_nll = (
        gaussian_nll_per_horizon(
            mean_t,
            scale_t,
            truth_t,
        )
        .numpy()
    )

    q = (
        mahalanobis_squared(
            mean_t,
            scale_t,
            truth_t,
        )
        .numpy()
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
                variance_scale
            ),
        )
    )

    calibrated_nll = (
        calibrated_nll_per_horizon(
            raw_nll,
            q,
            variance_scale,
        )
    )

    Gaussian_NLL = {
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

    # Direct calibrated-SPD proof.
    calibrated_scale_t = (
        apply_variance_scale(
            scale_t,
            variance_scale,
        )
    )

    calibrated_covariance = (
        calibrated_scale_t
        @
        calibrated_scale_t.transpose(
            -1,
            -2,
        )
    )

    (
        _,
        calibrated_info,
    ) = torch.linalg.cholesky_ex(
        calibrated_covariance
    )

    if bool(
        torch.any(
            calibrated_info != 0
        )
    ):
        raise RuntimeError(
            "Formal calibrated Gaussian "
            "covariance lost SPD."
        )

    # ========================================================
    # GMM formal metrics
    # ========================================================

    GMM_metrics = gmm_formal_metrics(
        merged[
            "gmm_logits"
        ],
        merged[
            "gmm_means"
        ],
        truth,
        validity,
    )

    GMM_mix_point = (
        trajectory_point_metrics(
            GMM_metrics[
                "mixture_mean"
            ],
            truth,
            validity,
            classes,
            class_id_to_name=(
                CLASS_ID_TO_NAME
            ),
        )
    )

    GMM_map_point = (
        trajectory_point_metrics(
            GMM_metrics[
                "MAP_mean"
            ],
            truth,
            validity,
            classes,
            class_id_to_name=(
                CLASS_ID_TO_NAME
            ),
        )
    )

    gmm_logits_t = torch.from_numpy(
        np.asarray(
            merged[
                "gmm_logits"
            ],
            dtype=np.float64,
        )
    )

    gmm_means_t = torch.from_numpy(
        np.asarray(
            merged[
                "gmm_means"
            ],
            dtype=np.float64,
        )
    )

    gmm_scale_t = torch.from_numpy(
        np.asarray(
            merged[
                "gmm_scale"
            ],
            dtype=np.float64,
        )
    )

    validity_t = torch.from_numpy(
        validity.astype(
            np.float64
        )
    )

    gmm_sample_nll = (
        gmm_joint_nll_per_sample(
            gmm_logits_t,
            gmm_means_t,
            gmm_scale_t,
            truth_t,
            validity_t,
        )
        .numpy()
    )

    active = validity.any(
        axis=1
    )

    GMM_joint_NLL = float(
        np.mean(
            gmm_sample_nll[
                active
            ]
        )
    )

    # ========================================================
    # Availability / association-side target recall
    # ========================================================

    # Use scenario-qualified namespaces, because raw WOMD
    # track indices repeat between scenarios.
    matched_namespace = (
        merge_summary[
            "matched_namespace"
        ]
    )

    eligible_namespace = (
        merge_summary[
            "eligible_namespace"
        ]
    )

    if (
        len(
            set(
                tuple(
                    item
                )
                for item in (
                    matched_namespace
                )
            )
        )
        !=
        len(
            matched_namespace
        )
    ):
        raise RuntimeError(
            "Duplicate matched formal "
            "scenario-track namespace."
        )

    matched_count = len(
        matched_namespace
    )

    eligible_count = len(
        eligible_namespace
    )

    matched_recall = (
        matched_count
        /
        eligible_count
    )

    availability = {
        "matched_tracks":
            matched_count,

        "eligible_truth_tracks":
            eligible_count,

        "matched_target_recall":
            float(
                matched_recall
            ),

        "horizons":
            {},
    }

    for h, horizon in enumerate(
        (
            "0.1",
            "0.3",
            "0.5",
            "1.0",
        )
    ):
        available = int(
            validity[
                :,
                h
            ].sum()
        )

        eligible = int(
            merged[
                "eligible_validity"
            ][
                :,
                h
            ].sum()
        )

        availability[
            "horizons"
        ][
            horizon
        ] = {
            "available_predictions":
                available,

            "eligible_truth_tracks":
                eligible,

            "availability_recall":
                (
                    float(
                        available
                        /
                        eligible
                    )
                    if eligible
                    else None
                ),
        }

    class_availability = {}

    for class_id in sorted(
        set(
            merged[
                "eligible_class_id"
            ].tolist()
        )
    ):
        class_id = int(
            class_id
        )

        name = CLASS_ID_TO_NAME.get(
            class_id,
            str(
                class_id
            ),
        )

        eligible = int(
            (
                merged[
                    "eligible_class_id"
                ]
                ==
                class_id
            ).sum()
        )

        matched = int(
            (
                classes
                ==
                class_id
            ).sum()
        )

        class_availability[
            name
        ] = {
            "eligible_truth_tracks":
                eligible,

            "matched_tracks":
                matched,

            "recall":
                float(
                    matched
                    /
                    eligible
                )
                if eligible
                else None,
        }

    availability[
        "classes"
    ] = class_availability

    # ========================================================
    # Exact frozen classical comparison
    # ========================================================

    classical = classical_metrics(
        Stage3_formal
    )

    if "CV" not in classical:
        raise RuntimeError(
            "Frozen Stage3 formal report "
            "does not expose CV metrics."
        )

    calibrated_Gaussian_ADE = float(
        Gaussian_point_metrics[
            "ade_m"
        ]
    )

    beaten = [
        name
        for name, metrics
        in classical.items()
        if (
            calibrated_Gaussian_ADE
            <
            metrics[
                "ade_m"
            ]
        )
    ]

    PDF_performance_gate = bool(
        beaten
    )

    internal_CV_gate = bool(
        calibrated_Gaussian_ADE
        <
        classical[
            "CV"
        ][
            "ade_m"
        ]
    )

    calibration_measured = True

    PDF_acceptance = (
        PDF_performance_gate
        and
        calibration_measured
    )

    raw_ECE = float(
        raw_reliability[
            "coverage_ECE_macro"
        ]
    )

    calibrated_ECE = float(
        calibrated_reliability[
            "coverage_ECE_macro"
        ]
    )

    formal_calibration_nonworsening = bool(
        calibrated_ECE
        <=
        raw_ECE
        +
        1e-15
    )

    # ========================================================
    # Final regression
    # ========================================================

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
        test_count != 114
    ):
        print(
            combined
        )

        raise RuntimeError(
            "Expected final Stage4 "
            "regression 114/114."
        )

    (
        implementation_files,
        implementation_sha,
    ) = implementation_fingerprint()

    free_gib = (
        shutil.disk_usage(
            ROOT
        ).free
        /
        1024**3
    )

    if free_gib < MIN_FREE_GIB:
        raise RuntimeError(
            "250-GiB storage reserve violated."
        )

    final_report = {
        "stage":
            4,

        "block":
            "4.8",

        "status":
            "PASS",

        "formal_population": {
            "scenario_count":
                120,

            "manifest_sha256":
                EXPECTED_FORMAL_MANIFEST_SHA,

            "Stage3_formal_run_sha256":
                EXPECTED_STAGE3_FORMAL_RUN_SHA,

            "same_frozen_population":
                True,
        },

        "leakage_contract": {
            "training_on_formal":
                False,

            "model_selection_on_formal":
                False,

            "normalization_fit_on_formal":
                False,

            "calibrator_fit_on_formal":
                False,

            "tracks_to_predict_model_input":
                False,

            "tracks_to_predict_accessed_after_prediction":
                True,

            "future_model_input":
                False,
        },

        "observation_interface": {
            "Stage2":
                "frozen_full_degraded",

            "association":
                "frozen_Stage3_estimated_GNN",

            "measurement_covariance_R_t":
                "real_input",

            "formal_cache_resumable":
                True,
        },

        "availability":
            availability,

        "neural_models": {
            "deterministic_GRU":
                deterministic_metrics,

            "raw_Gaussian_GRU": {
                "point_metrics":
                    Gaussian_point_metrics,

                "NLL":
                    Gaussian_NLL[
                        "raw"
                    ],

                "reliability":
                    raw_reliability,
            },

            "calibrated_Gaussian_GRU": {
                "point_metrics":
                    Gaussian_point_metrics,

                "mean_identical_to_raw":
                    True,

                "NLL":
                    Gaussian_NLL[
                        "calibrated"
                    ],

                "reliability":
                    calibrated_reliability,

                "covariance_SPD":
                    True,

                "calibrator_sha256":
                    EXPECTED_CALIBRATOR_SHA,
            },

            "GMM_ablation": {
                "mixture_mean_point_metrics":
                    GMM_mix_point,

                "MAP_point_metrics":
                    GMM_map_point,

                "joint_GMM_NLL_per_sample":
                    GMM_joint_NLL,

                "minADE_m":
                    GMM_metrics[
                        "minADE_m"
                    ],

                "minFDE_1.0s_m":
                    GMM_metrics[
                        "minFDE_1.0s_m"
                    ],

                "mixture_entropy_nats":
                    GMM_metrics[
                        "mixture_entropy_nats"
                    ],

                "effective_mode_count":
                    GMM_metrics[
                        "effective_mode_count"
                    ],

                "mean_mixture_probability":
                    GMM_metrics[
                        "mean_mixture_probability"
                    ],

                "MAP_mode_counts":
                    GMM_metrics[
                        "MAP_mode_counts"
                    ],

                "calibrated":
                    False,

                "downstream_selected":
                    False,
            },
        },

        "classical_frozen_Stage3":
            classical,

        "comparison": {
            "primary_metric":
                "planar_ADE_m",

            "calibrated_Gaussian_ADE_m":
                calibrated_Gaussian_ADE,

            "classical_baselines_beaten":
                beaten,

            "beats_at_least_one_classical":
                PDF_performance_gate,

            "frozen_CV_ADE_m":
                classical[
                    "CV"
                ][
                    "ade_m"
                ],

            "internal_CV_gate":
                internal_CV_gate,
        },

        "formal_calibration_generalization": {
            "raw_macro_ECE":
                raw_ECE,

            "calibrated_macro_ECE":
                calibrated_ECE,

            "ECE_nonworsening":
                formal_calibration_nonworsening,

            "raw_coverage_event_Brier":
                raw_reliability[
                    "coverage_event_Brier"
                ],

            "calibrated_coverage_event_Brier":
                calibrated_reliability[
                    "coverage_event_Brier"
                ],

            "raw_metric_Gaussian_NLL":
                Gaussian_NLL[
                    "raw"
                ][
                    "metric_Gaussian_NLL"
                ],

            "calibrated_metric_Gaussian_NLL":
                Gaussian_NLL[
                    "calibrated"
                ][
                    "metric_Gaussian_NLL"
                ],
        },

        "Stage4_acceptance_evidence": {
            "PDF_probabilistic_beats_at_least_one_classical":
                PDF_performance_gate,

            "PDF_calibration_measured":
                calibration_measured,

            "PDF_acceptance_current":
                PDF_acceptance,

            "internal_CV_gate":
                internal_CV_gate,

            "final_closure_decision":
                "DEFERRED_TO_BLOCK4.10",
        },

        "reproducibility": {
            "formal_inference_repeat_scenes":
                4,

            "exact_repeat":
                True,

            "merged_formal_content_sha256":
                merge_summary[
                    "merged_content_sha256"
                ],

            "merged_formal_file_sha256":
                merge_summary[
                    "merged_file_sha256"
                ],
        },

        "regression": {
            "tests_passed":
                114,

            "tests_total":
                114,
        },

        "implementation": {
            "file_count":
                implementation_files,

            "sha256":
                implementation_sha,
        },

        "storage": {
            "free_gib":
                free_gib,

            "reserve_gib":
                MIN_FREE_GIB,

            "pass":
                True,
        },
    }

    write_json(
        REPORT,
        final_report,
    )

    if FAILURE.exists():
        FAILURE.unlink()

    marker = (
        "## Block 4.8 — "
        "Frozen formal N=120 evaluation"
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
                "Status: PASS / FROZEN EVALUATION\n\n"
                "- The immutable Stage3 formal "
                  "N=120 manifest is reused exactly.\n"
                "- The same frozen full-degraded "
                  "Stage2 observation interface and "
                  "Stage3 estimated-GNN association "
                  "feed all Stage4 neural models.\n"
                "- No formal scenario contributes to "
                  "training, normalization, model "
                  "selection or calibration fitting.\n"
                "- All causal neural predictions are "
                  "generated before tracks_to_predict "
                  "is accessed for evaluator filtering.\n"
                "- Formal ADE follows Stage3's "
                  "available matched actor-horizon "
                  "micro-mean semantics; target "
                  "availability/recall is reported "
                  "separately.\n"
                "- Deterministic GRU, raw Gaussian, "
                  "calibrated Gaussian and GMM are "
                  "evaluated on the same neural target "
                  "population.\n"
                "- Raw and calibrated Gaussian NLL, "
                  "50/80/90/95/99 coverage, coverage "
                  "ECE and coverage-event Brier are "
                  "measured without recalibration.\n"
                "- GMM remains an uncalibrated "
                  "multimodal ablation and is not "
                  "selected downstream here.\n"
                "- Classical comparisons reuse the "
                  "exact frozen Stage3 formal report; "
                  "classical algorithms are not rerun.\n"
                f"- Calibrated Gaussian formal ADE: "
                f"{calibrated_Gaussian_ADE:.9f} m.\n"
                f"- Classical baselines beaten: "
                f"{beaten}.\n"
                f"- PDF performance gate: "
                f"{PDF_performance_gate}.\n"
                f"- Internal CV gate: "
                f"{internal_CV_gate}.\n"
                f"- Raw formal macro ECE: "
                f"{raw_ECE:.9f}.\n"
                f"- Calibrated formal macro ECE: "
                f"{calibrated_ECE:.9f}.\n"
                f"- Formal ECE non-worsening: "
                f"{formal_calibration_nonworsening}.\n"
                "- Exact repeated formal inference: "
                  "4/4 PASS.\n"
                "- Full Stage4 regression: "
                  "114/114 PASS.\n"
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
        "STAGE4 BLOCK 4.8 FORMAL GATE"
    )
    print(
        "============================================================"
    )

    print(
        "formal scenarios              = 120 / 120"
    )
    print(
        "formal manifest SHA           = PASS"
    )
    print(
        "Stage3 formal run SHA         = PASS"
    )
    print(
        "tracks_to_predict after pred  = PASS"
    )
    print(
        "formal training               = NO"
    )
    print(
        "formal model selection        = NO"
    )
    print(
        "formal recalibration          = NO"
    )

    print()
    print(
        "TARGET AVAILABILITY"
    )
    print(
        "eligible formal targets      =",
        eligible_count,
    )
    print(
        "matched neural targets       =",
        matched_count,
    )
    print(
        "matched-target recall        =",
        round(
            matched_recall,
            6,
        ),
    )

    print()
    print(
        "NEURAL FORMAL METRICS"
    )
    print(
        "deterministic ADE m          =",
        round(
            deterministic_metrics[
                "ade_m"
            ],
            6,
        ),
    )
    print(
        "Gaussian ADE m               =",
        round(
            Gaussian_point_metrics[
                "ade_m"
            ],
            6,
        ),
    )
    print(
        "Gaussian 1.0s FDE m          =",
        round(
            Gaussian_point_metrics[
                "fde_1.0s_m"
            ],
            6,
        ),
    )
    print(
        "GMM mixture ADE m            =",
        round(
            GMM_mix_point[
                "ade_m"
            ],
            6,
        ),
    )
    print(
        "GMM MAP ADE m                =",
        round(
            GMM_map_point[
                "ade_m"
            ],
            6,
        ),
    )
    print(
        "GMM minADE m                 =",
        round(
            GMM_metrics[
                "minADE_m"
            ],
            6,
        ),
    )
    print(
        "GMM minFDE @1.0s m           =",
        round(
            GMM_metrics[
                "minFDE_1.0s_m"
            ],
            6,
        ),
    )

    print()
    print(
        "FORMAL GAUSSIAN CALIBRATION"
    )
    print(
        "raw NLL                     =",
        round(
            Gaussian_NLL[
                "raw"
            ][
                "metric_Gaussian_NLL"
            ],
            6,
        ),
    )
    print(
        "calibrated NLL              =",
        round(
            Gaussian_NLL[
                "calibrated"
            ][
                "metric_Gaussian_NLL"
            ],
            6,
        ),
    )
    print(
        "raw macro ECE               =",
        round(
            raw_ECE,
            6,
        ),
    )
    print(
        "calibrated macro ECE        =",
        round(
            calibrated_ECE,
            6,
        ),
    )
    print(
        "formal ECE non-worsening    =",
        (
            "PASS"
            if formal_calibration_nonworsening
            else
            "NO"
        ),
    )
    print(
        "raw Brier                   =",
        round(
            raw_reliability[
                "coverage_event_Brier"
            ],
            6,
        ),
    )
    print(
        "calibrated Brier            =",
        round(
            calibrated_reliability[
                "coverage_event_Brier"
            ],
            6,
        ),
    )

    print()
    print(
        "CALIBRATED FORMAL COVERAGE"
    )

    for level in (
        "0.50",
        "0.80",
        "0.90",
        "0.95",
        "0.99",
    ):
        empirical = (
            calibrated_reliability[
                "global_levels"
            ][
                level
            ][
                "empirical"
            ]
        )

        print(
            f"coverage {int(float(level)*100):02d}%"
            f"                    = "
            f"{empirical:.6f}"
        )

    print()
    print(
        "FROZEN CLASSICAL ADE"
    )

    for name, metrics in sorted(
        classical.items(),
        key=lambda item:
            item[
                1
            ][
                "ade_m"
            ],
    ):
        print(
            f"{name:12s} = "
            f"{metrics['ade_m']:.6f} m"
        )

    print()
    print(
        "probabilistic baselines beaten =",
        beaten,
    )
    print(
        "PDF beats >=1 classical       =",
        (
            "PASS"
            if PDF_performance_gate
            else
            "BLOCKED"
        ),
    )
    print(
        "internal Gaussian < CV gate   =",
        (
            "PASS"
            if internal_CV_gate
            else
            "BLOCKED"
        ),
    )
    print(
        "calibration measured          = PASS"
    )
    print(
        "PDF Stage4 evidence current   =",
        (
            "PASS"
            if PDF_acceptance
            else
            "BLOCKED"
        ),
    )

    print()
    print(
        "formal inference repeat       = 4 / 4 PASS"
    )
    print(
        "formal output content SHA     =",
        merge_summary[
            "merged_content_sha256"
        ],
    )
    print(
        "full Stage4 regression        = 114 / 114 PASS"
    )
    print(
        "free GiB                      =",
        round(
            free_gib,
            3,
        ),
    )
    print(
        "implementation files          =",
        implementation_files,
    )
    print(
        "implementation SHA256         =",
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
        "===== BLOCK 4.8 FINAL ====="
    )
    print(
        "frozen formal N=120           = EVALUATED"
    )
    print(
        "same degraded interface       = PASS"
    )
    print(
        "deterministic GRU             = EVALUATED"
    )
    print(
        "raw Gaussian                  = EVALUATED"
    )
    print(
        "calibrated Gaussian           = EVALUATED"
    )
    print(
        "GMM                           = EVALUATED"
    )
    print(
        "class/horizon metrics         = MEASURED"
    )
    print(
        "formal calibration            = MEASURED"
    )
    print(
        "classical comparison          = FROZEN REPORT"
    )
    print(
        "formal leakage                = NONE"
    )
    print(
        "checkpoint/model selection    = NONE"
    )
    print(
        "regression                    = 114 / 114"
    )
    print(
        "implementation SHA            =",
        implementation_sha,
    )
    print(
        "log closure                   = PASS"
    )


def recovery_hint(
    exc,
):
    text = (
        f"{type(exc).__name__}: "
        f"{exc}"
    ).lower()

    if (
        "validation manifest"
        in text
    ):
        return (
            "Do not guess the validation reader. "
            "Inspect the candidate manifest evidence; "
            "no model retraining is needed."
        )

    if (
        "truth-track"
        in text
        or
        "truth_track"
        in text
        or
        "sample"
        in text
    ):
        return (
            "Inspect the schema probe / explicit "
            "supervision metadata. Do not use "
            "prediction_id or tracks_to_predict "
            "as a causal identity feature."
        )

    if (
        "checkpoint"
        in text
        or
        "state"
        in text
        or
        "sha"
        in text
    ):
        return (
            "Do not retrain or recalibrate. "
            "Audit the frozen Stage3/Stage4 "
            "artifact hash that failed."
        )

    if (
        "cuda"
        in text
        or
        "out of memory"
        in text
    ):
        return (
            "Check nvidia-smi first. Valid formal "
            "per-scenario shards are resumable."
        )

    if (
        "114/114"
        in text
        or
        "regression"
        in text
    ):
        return (
            "Inspect the failing regression. "
            "Do not modify frozen Blocks4.0–4.7."
        )

    return (
        "Inspect reports/"
        "block48_formal_evaluation_failure.json. "
        "Completed formal scenario shards are "
        "resumable; no retraining/recalibration "
        "or Stage3 classical rerun is required."
    )


try:
    main()

except BaseException as exc:
    payload = {
        "stage":
            4,

        "block":
            "4.8_part_2",

        "status":
            "BLOCKED",

        "exception_type":
            type(
                exc
            ).__name__,

        "exception":
            str(
                exc
            ),

        "traceback":
            traceback.format_exc(),

        "recovery_hint":
            recovery_hint(
                exc
            ),

        "formal_cache_resumable":
            True,

        "training_performed":
            False,

        "model_selection_performed":
            False,

        "calibrator_fit_performed":
            False,

        "Stage3_classical_rerun":
            False,

        "upstream_modified":
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
        "BLOCK 4.8 PART 2/2 = BLOCKED"
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
        "formal cache reusable      = YES"
    )
    print(
        "training performed         = NO"
    )
    print(
        "model selection performed  = NO"
    )
    print(
        "recalibration performed    = NO"
    )
    print(
        "Stage3 classical rerun     = NO"
    )
    print(
        "terminal remains open      = YES"
    )
    print(
        "failure report =",
        FAILURE,
    )

# Deliberately no sys.exit().

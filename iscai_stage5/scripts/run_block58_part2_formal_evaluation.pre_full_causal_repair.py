from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch


# ============================================================
# BLOCK 5.8 PART 2/2
# Frozen formal Stage-5 N=120 evaluation.
#
# HARD SCIENTIFIC RULES:
# - same immutable Stage3/4 formal population
# - no training / refit / recalibration
# - no tracks_to_predict receiver selection
# - no future truth in controller
# - primary receiver: nearest causal vehicle ahead
# - frozen calibrated Gaussian Stage4 posterior
# - centroid / known / uncertain receiver geometry
# - 16 / 32 / 64 codebooks
# - exhaustive / persistence / geometry-nearest /
#   fixed Top-1/3/5 / oracle
# - adaptive Top-K at 0.90 / 0.95 / 0.975 / 0.99
# - future truth only after controller decisions
# - frozen q=0.95 formal acceptance
# - no post-hoc tuning
#
# Exit:
#   0 = implementation + formal acceptance PASS
#   2 = implementation/provenance/runtime failure
#   3 = formal evaluation completed but scientific gate BLOCKED
# ============================================================


ROOT = Path("/home/agni/waymo")
S3 = ROOT / "iscai_stage3"
S4 = ROOT / "iscai_stage4"
S5 = ROOT / "iscai_stage5"

FORMAL_MANIFEST = (
    S3
    / "artifacts/block38e/formal_validation_120.jsonl"
)

EXPECTED_FORMAL_MANIFEST_SHA = (
    "2208e7287ddf6439fda4597c435a9cba"
    "1d1b9d0e4c4547bc5dd92e56e8124e46"
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

GAUSSIAN_CHECKPOINT = (
    S4
    / "artifacts/block44/gaussian_gru.pt"
)

EXPECTED_GAUSSIAN_SHA = (
    "49ff64d145eaa633f295c16f660df380"
    "c35383e7e3b61279a5aad7cd700d619f"
)

NORMALIZATION = (
    S4
    / "artifacts/block43/fit_normalization.json"
)

EXPECTED_NORMALIZATION_SHA = (
    "3d7fc0a66d4a4f566f6569befa9c3766"
    "ecae21830bb4e46256df2f326a82a5f6"
)

CALIBRATOR = (
    S4
    / "artifacts/block45/covariance_scaler.json"
)

EXPECTED_CALIBRATOR_SHA = (
    "508ff2e3fbcfafe8e001155340c25baaf"
    "3772fe2561a8022a9ed1cf780e66087"
)

STAGE4_HANDOFF = (
    S4
    / "artifacts/block410/stage4_to_stage5_handoff.json"
)

EXPECTED_STAGE4_HANDOFF_SHA = (
    "491bce010d35c2a394f879ecf35ed26e"
    "f1de92f7fff1465dcbf46b072a87c6fd"
)

ACCEPTANCE_CONFIG = (
    S5
    / "configs/formal_stage5_acceptance_policy.json"
)

EXPECTED_ACCEPTANCE_SHA = (
    "a89dcc3c785933632b547d73dae96696"
    "e3b63ebd19b46e73b7b0602fa0da313f"
)

PREFREEZE_REPORT = (
    S5
    / "reports/block58_prefreeze_acceptance.json"
)

LATENCY_CONFIG = (
    S5
    / "configs/beam_latency_policy.json"
)

RECEIVER_CONFIG = (
    S5
    / "configs/receiver_policy.json"
)

ANGULAR_CONFIG = (
    S5
    / "configs/angular_posterior_policy.json"
)

BLOCK44_TRAINING_SCRIPT = (
    S4
    / "scripts/run_block44_gaussian_training.py"
)

CACHE_DIR = (
    S5
    / "artifacts/block58/formal_stage5_cache"
)

CACHE_MANIFEST = (
    S5
    / "artifacts/block58/formal_stage5_cache_manifest.json"
)

MERGED_RECORDS = (
    S5
    / "artifacts/block58/formal_stage5_records.jsonl"
)

REPORT = (
    S5
    / "reports/block58_formal_evaluation.json"
)

FAILURE_REPORT = (
    S5
    / "reports/block58_formal_failure.json"
)

MIN_FREE_BYTES = 250 * (1024 ** 3)

FORMAL_N = 120

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

CODEBOOK_SIZES = (
    16,
    32,
    64,
)

REQUESTED_Q = (
    0.90,
    0.95,
    0.975,
    0.99,
)

NOMINAL_Q = 0.95

FIXED_TOPK = (
    1,
    3,
    5,
)

GEOMETRY_MODES = (
    "centroid_baseline",
    "known_receiver_offset",
    "uncertain_receiver_offset",
)

# Primary full uncertainty-aware Stage5 case.
# Centroid and known-offset remain mandatory comparisons.
PRIMARY_FORMAL_GEOMETRY = (
    "uncertain_receiver_offset"
)

BASE_SEED = 20260821

# Descriptive-only fixed slicing.
# Never used for policy choice or acceptance.
DISTANCE_BINS_M = (
    0.0,
    25.0,
    50.0,
    100.0,
    float("inf"),
)

UNCERTAINTY_BINS_DEG = (
    0.0,
    1.0,
    3.0,
    5.0,
    float("inf"),
)


# ============================================================
# Frozen imports
# ============================================================

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

from iscai_stage4.ml.gaussian_gru import (
    GaussianTrajectoryGRU,
)

from iscai_stage4.ml.calibration import (
    apply_variance_scale,
)

from iscai_stage4.ml.formal_runtime import (
    resolve_sample_truth_track_index,
)

from iscai_stage5.receiver_selection import (
    ReceiverCandidate,
    ReceiverSelectionConfig,
    normalize_semantic_class,
    select_primary_receiver,
)

from iscai_stage5.receiver_geometry import (
    ReceiverOffsetDistribution,
    receiver_geometry_distribution_h0,
)

from iscai_stage5.angular_monte_carlo import (
    ReceiverOffsetGaussianBody,
    deterministic_receiver_angular_posterior,
)

from iscai_stage5.beam_codebook import (
    beam_probability_mass_from_samples,
    build_uniform_azimuth_codebook,
)

from iscai_stage5.beam_baselines import (
    fixed_top_k_probability,
    geometry_nearest_beam,
    oracle_best_gain_beam,
)

from iscai_stage5.adaptive_topk_temporal import (
    AdaptiveTemporalDecision,
    AdaptiveTemporalState,
    adaptive_temporal_step,
    initial_adaptive_temporal_state,
)

from iscai_stage5.optical_link import (
    evaluate_optical_link,
    optical_gain_for_cell,
)

from iscai_stage5.beam_latency import (
    actual_probe_count_for_adaptive_decision,
)

from iscai_stage5.formal_acceptance import (
    FORMAL_EMPIRICAL_ALPHA,
    empirical_coverage_decision,
)


# ============================================================
# Utilities
# ============================================================

def file_sha256(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as stream:
        while True:
            block = stream.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    data = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    return sha256(data).hexdigest()


def atomic_json(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def atomic_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
            stream.write("\n")

    os.replace(
        temporary,
        path,
    )


def load_json(
    path: Path,
) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def mean_or_none(
    values,
):
    values = [
        float(value)
        for value in values
        if value is not None
        and math.isfinite(
            float(value)
        )
    ]

    if not values:
        return None

    return float(
        np.mean(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )
    )


def wrap_angle(
    value: float,
) -> float:
    return float(
        (
            float(value)
            +
            math.pi
        )
        %
        (
            2.0
            *
            math.pi
        )
        -
        math.pi
    )


def stable_seed(
    *parts,
) -> int:
    text = "|".join(
        str(part)
        for part in parts
    )

    value = int(
        sha256(
            text.encode("utf-8")
        ).hexdigest()[
            :8
        ],
        16,
    )

    return int(
        (
            BASE_SEED
            +
            value
        )
        %
        (
            2**31 - 1
        )
    )


def recursive_values_for_key(
    value,
    key_name: str,
    prefix="",
):
    out = []

    if isinstance(value, dict):
        for key, item in value.items():
            path = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            if str(key) == key_name:
                out.append(
                    (
                        path,
                        item,
                    )
                )

            out.extend(
                recursive_values_for_key(
                    item,
                    key_name,
                    path,
                )
            )

    elif isinstance(value, list):
        for index, item in enumerate(
            value
        ):
            out.extend(
                recursive_values_for_key(
                    item,
                    key_name,
                    f"{prefix}[{index}]",
                )
            )

    return out


def recursive_path_values(
    value,
    prefix="",
):
    if isinstance(value, dict):
        for key, item in value.items():
            path = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            yield from recursive_path_values(
                item,
                path,
            )

    elif isinstance(value, list):
        yield (
            prefix,
            value,
        )

        for index, item in enumerate(
            value
        ):
            yield from recursive_path_values(
                item,
                f"{prefix}[{index}]",
            )

    else:
        yield (
            prefix,
            value,
        )


def unique_numeric_key(
    value,
    key_name: str,
    *,
    default=None,
):
    found = []

    for _, item in (
        recursive_values_for_key(
            value,
            key_name,
        )
    ):
        if (
            isinstance(
                item,
                (int, float),
            )
            and
            not isinstance(
                item,
                bool,
            )
            and
            math.isfinite(
                float(item)
            )
        ):
            found.append(
                float(item)
            )

    if not found:
        if default is not None:
            return float(default)

        raise RuntimeError(
            f"Could not resolve frozen {key_name}."
        )

    unique = sorted(
        set(
            found
        )
    )

    if len(unique) > 1:
        raise RuntimeError(
            f"Ambiguous {key_name}: {unique}"
        )

    return float(
        unique[
            0
        ]
    )


def verify_exact_sha(
    path: Path,
    expected: str,
    label: str,
):
    if not path.is_file():
        raise RuntimeError(
            f"{label} missing: {path}"
        )

    actual = file_sha256(
        path
    )

    if actual != expected:
        raise RuntimeError(
            f"{label} SHA changed. "
            f"expected={expected} actual={actual}"
        )


def strip_nondeterministic_runtime(
    value,
):
    if isinstance(value, dict):
        result = {}

        for key, item in value.items():
            lower = str(
                key
            ).lower()

            if (
                lower.endswith(
                    "_runtime_s"
                )
                or
                lower.endswith(
                    "_wall_s"
                )
                or
                lower.endswith(
                    "_elapsed_s"
                )
            ):
                continue

            result[
                key
            ] = (
                strip_nondeterministic_runtime(
                    item
                )
            )

        return result

    if isinstance(value, list):
        return [
            strip_nondeterministic_runtime(
                item
            )
            for item in value
        ]

    return value


# ============================================================
# Preflight provenance
# ============================================================

def read_formal_rows():
    rows = [
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
    ]

    if len(rows) != FORMAL_N:
        raise RuntimeError(
            f"Formal manifest count changed: "
            f"{len(rows)}"
        )

    scenario_ids = [
        str(
            row[
                "scenario_id"
            ]
        )
        for row in rows
    ]

    if len(
        set(
            scenario_ids
        )
    ) != FORMAL_N:
        raise RuntimeError(
            "Formal scenario IDs are not unique."
        )

    return rows


def verify_prefreeze():
    verify_exact_sha(
        FORMAL_MANIFEST,
        EXPECTED_FORMAL_MANIFEST_SHA,
        "formal manifest",
    )

    verify_exact_sha(
        GAUSSIAN_CHECKPOINT,
        EXPECTED_GAUSSIAN_SHA,
        "Gaussian checkpoint",
    )

    verify_exact_sha(
        NORMALIZATION,
        EXPECTED_NORMALIZATION_SHA,
        "fit-only normalization",
    )

    verify_exact_sha(
        CALIBRATOR,
        EXPECTED_CALIBRATOR_SHA,
        "covariance calibrator",
    )

    verify_exact_sha(
        STAGE4_HANDOFF,
        EXPECTED_STAGE4_HANDOFF_SHA,
        "Stage4->Stage5 handoff",
    )

    verify_exact_sha(
        ACCEPTANCE_CONFIG,
        EXPECTED_ACCEPTANCE_SHA,
        "frozen Stage5 acceptance config",
    )

    prefreeze = load_json(
        PREFREEZE_REPORT
    )

    if (
        prefreeze.get(
            "status"
        )
        !=
        "PASS_PREFORMAL_FREEZE"
    ):
        raise RuntimeError(
            "Block5.8 pre-formal freeze is not PASS."
        )

    if bool(
        prefreeze.get(
            "formal_stage5_metrics_computed"
        )
    ):
        raise RuntimeError(
            "Pre-freeze report claims formal metrics "
            "were already computed."
        )

    acceptance = load_json(
        ACCEPTANCE_CONFIG
    )

    if (
        acceptance.get(
            "status"
        )
        !=
        "FROZEN_PRE_FORMAL"
    ):
        raise RuntimeError(
            "Formal acceptance policy is not frozen."
        )

    if (
        float(
            acceptance[
                "coverage_targets"
            ][
                "nominal_primary_q"
            ]
        )
        !=
        NOMINAL_Q
    ):
        raise RuntimeError(
            "Frozen nominal q changed."
        )

    if tuple(
        float(value)
        for value in (
            acceptance[
                "coverage_targets"
            ][
                "reported_q"
            ]
        )
    ) != REQUESTED_Q:
        raise RuntimeError(
            "Frozen requested-q set changed."
        )

    if tuple(
        int(value)
        for value in (
            acceptance[
                "evaluation_granularity"
            ][
                "codebook_sizes"
            ]
        )
    ) != CODEBOOK_SIZES:
        raise RuntimeError(
            "Frozen codebook sizes changed."
        )

    if tuple(
        float(value)
        for value in (
            acceptance[
                "evaluation_granularity"
            ][
                "horizons_s"
            ]
        )
    ) != HORIZONS_S:
        raise RuntimeError(
            "Frozen formal horizons changed."
        )

    if (
        float(
            acceptance[
                "formal_empirical_coverage_tolerance"
            ][
                "alpha"
            ]
        )
        !=
        FORMAL_EMPIRICAL_ALPHA
    ):
        raise RuntimeError(
            "Frozen formal alpha changed."
        )

    free_bytes = shutil.disk_usage(
        S5
    ).free

    if free_bytes < MIN_FREE_BYTES:
        raise RuntimeError(
            "Storage reserve below frozen "
            "250 GiB requirement."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for frozen "
            "Stage4 Gaussian inference."
        )

    return acceptance


# ============================================================
# Frozen Stage4 Gaussian runtime
# ============================================================

def load_block44_normalizer_class():
    if not BLOCK44_TRAINING_SCRIPT.is_file():
        raise RuntimeError(
            "Frozen Block4.4 training script missing."
        )

    spec = (
        importlib.util
        .spec_from_file_location(
            "_iscai_block44_runtime",
            BLOCK44_TRAINING_SCRIPT,
        )
    )

    if (
        spec is None
        or
        spec.loader is None
    ):
        raise RuntimeError(
            "Could not load Block4.4 runtime module."
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    sys.modules[
        spec.name
    ] = module

    spec.loader.exec_module(
        module
    )

    cls = getattr(
        module,
        "GaussianNormalizer",
        None,
    )

    if cls is None:
        raise RuntimeError(
            "GaussianNormalizer not found in "
            "frozen Block4.4 script."
        )

    return cls


def gaussian_architecture_kwargs(
    configuration,
):
    defaults = {
        "target_hidden_dim":
            64,

        "neighbor_hidden_dim":
            32,

        "map_hidden_dim":
            32,

        "fusion_hidden_dim":
            128,
    }

    output = {}

    for key, default in defaults.items():
        values = []

        for _, item in (
            recursive_values_for_key(
                configuration,
                key,
            )
        ):
            if (
                isinstance(
                    item,
                    int,
                )
                and
                not isinstance(
                    item,
                    bool,
                )
            ):
                values.append(
                    int(
                        item
                    )
                )

        if values:
            unique = sorted(
                set(
                    values
                )
            )

            if len(unique) != 1:
                raise RuntimeError(
                    f"Ambiguous frozen architecture "
                    f"{key}: {unique}"
                )

            output[
                key
            ] = unique[
                0
            ]

        else:
            # These are the already frozen 4.3/4.4
            # architecture values, not a new choice.
            output[
                key
            ] = default

    return output


def variance_scale_alpha_h(
    calibrator,
):
    candidates = []

    for path, value in (
        recursive_path_values(
            calibrator
        )
    ):
        lower = path.lower()

        if not (
            "scale" in lower
            or
            "alpha" in lower
        ):
            continue

        if (
            isinstance(
                value,
                list,
            )
            and
            len(
                value
            )
            ==
            4
            and
            all(
                isinstance(
                    item,
                    (int, float),
                )
                and
                not isinstance(
                    item,
                    bool,
                )
                and
                math.isfinite(
                    float(
                        item
                    )
                )
                and
                float(
                    item
                )
                >
                0.0
                for item in value
            )
        ):
            candidates.append(
                (
                    path,
                    tuple(
                        float(
                            item
                        )
                        for item in value
                    ),
                )
            )

    preferred = [
        candidate
        for candidate in candidates
        if (
            "variance" in candidate[
                0
            ].lower()
            or
            "alpha_h" in candidate[
                0
            ].lower()
        )
    ]

    source = (
        preferred
        if preferred
        else candidates
    )

    if not source:
        raise RuntimeError(
            "Could not resolve frozen variance "
            "calibration alpha_h."
        )

    unique = sorted(
        set(
            value
            for _, value in source
        )
    )

    if len(unique) != 1:
        raise RuntimeError(
            "Ambiguous covariance calibration "
            f"scales: {source}"
        )

    return (
        unique[
            0
        ],
        [
            path
            for path, value in source
            if value
            ==
            unique[
                0
            ]
        ][
            0
        ],
    )


@dataclass
class FrozenGaussianRuntime:
    model: GaussianTrajectoryGRU
    normalizer: Any
    variance_scale: tuple[float, ...]
    variance_scale_source: str
    device: torch.device


def build_gaussian_runtime():
    torch.manual_seed(
        BASE_SEED
    )

    np.random.seed(
        BASE_SEED
    )

    torch.use_deterministic_algorithms(
        True
    )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device(
        "cuda:0"
    )

    checkpoint = torch.load(
        GAUSSIAN_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise RuntimeError(
            "Gaussian checkpoint is not a mapping."
        )

    if (
        "state_dict"
        not in checkpoint
        or
        "configuration"
        not in checkpoint
    ):
        raise RuntimeError(
            "Gaussian checkpoint contract changed."
        )

    kwargs = (
        gaussian_architecture_kwargs(
            checkpoint[
                "configuration"
            ]
        )
    )

    model = (
        GaussianTrajectoryGRU(
            **kwargs,
            use_neighbors=True,
            use_map=True,
        )
        .to(
            device
        )
    )

    model.load_state_dict(
        checkpoint[
            "state_dict"
        ],
        strict=True,
    )

    model.eval()

    statistics = load_json(
        NORMALIZATION
    )

    Normalizer = (
        load_block44_normalizer_class()
    )

    normalizer = Normalizer(
        statistics,
        device,
    )

    calibrator = load_json(
        CALIBRATOR
    )

    if (
        str(
            calibrator.get(
                "Gaussian_checkpoint_sha256"
            )
        )
        !=
        EXPECTED_GAUSSIAN_SHA
    ):
        raise RuntimeError(
            "Calibrator does not reference "
            "the frozen Gaussian checkpoint."
        )

    variance_scale, source = (
        variance_scale_alpha_h(
            calibrator
        )
    )

    return FrozenGaussianRuntime(
        model=model,
        normalizer=normalizer,
        variance_scale=variance_scale,
        variance_scale_source=source,
        device=device,
    )


# ============================================================
# Stage5 frozen config resolution
# ============================================================

def find_receiver_offset():
    candidate_files = list(
        sorted(
            (S5 / "configs")
            .glob("*.json")
        )
    )

    candidate_files.extend(
        sorted(
            (ROOT / "iscai_stage1/configs")
            .glob("*.json")
        )
        if
        (
            ROOT
            / "iscai_stage1/configs"
        ).is_dir()
        else
        []
    )

    candidates = []

    def walk(
        value,
        path,
        source_path,
    ):
        if isinstance(
            value,
            dict,
        ):
            mean_key = None
            cov_key = None

            for key in value:
                lower = str(
                    key
                ).lower()

                if (
                    "mean" in lower
                    and
                    "body" in lower
                    and
                    (
                        "offset" in lower
                        or
                        lower.endswith(
                            "_m"
                        )
                    )
                ):
                    mean_key = key

                if (
                    "covariance" in lower
                    and
                    "body" in lower
                ):
                    cov_key = key

            if (
                mean_key is not None
                and
                cov_key is not None
            ):
                try:
                    mean = np.asarray(
                        value[
                            mean_key
                        ],
                        dtype=np.float64,
                    )

                    covariance = np.asarray(
                        value[
                            cov_key
                        ],
                        dtype=np.float64,
                    )

                    if (
                        mean.shape
                        ==
                        (
                            3,
                        )
                        and
                        covariance.shape
                        ==
                        (
                            3,
                            3,
                        )
                        and
                        np.all(
                            np.isfinite(
                                mean
                            )
                        )
                        and
                        np.all(
                            np.isfinite(
                                covariance
                            )
                        )
                    ):
                        score = 0

                        text = (
                            str(
                                source_path
                            )
                            +
                            ":"
                            +
                            path
                        ).lower()

                        if "receiver" in text:
                            score += 5

                        if "geometry" in text:
                            score += 5

                        if "stage5" in text:
                            score += 2

                        candidates.append(
                            (
                                -score,
                                str(
                                    source_path
                                ),
                                path,
                                mean,
                                covariance,
                            )
                        )
                except Exception:
                    pass

            for key, item in value.items():
                child = (
                    f"{path}.{key}"
                    if path
                    else str(key)
                )

                walk(
                    item,
                    child,
                    source_path,
                )

        elif isinstance(
            value,
            list,
        ):
            for index, item in enumerate(
                value
            ):
                walk(
                    item,
                    f"{path}[{index}]",
                    source_path,
                )

    for path in candidate_files:
        try:
            walk(
                load_json(
                    path
                ),
                "",
                path,
            )
        except Exception:
            continue

    if not candidates:
        raise RuntimeError(
            "Could not resolve frozen receiver "
            "offset mean/covariance."
        )

    candidates.sort(
        key=lambda item: (
            item[
                0
            ],
            item[
                1
            ],
            item[
                2
            ],
        )
    )

    _, source_path, key_path, mean, covariance = (
        candidates[
            0
        ]
    )

    covariance = (
        0.5
        *
        (
            covariance
            +
            covariance.T
        )
    )

    eigenvalues = np.linalg.eigvalsh(
        covariance
    )

    if np.min(
        eigenvalues
    ) < -1e-10:
        raise RuntimeError(
            "Frozen receiver placement covariance "
            "is not positive semidefinite."
        )

    return {
        "mean_body_m":
            tuple(
                float(
                    value
                )
                for value in mean
            ),

        "covariance_body_m2":
            tuple(
                tuple(
                    float(
                        value
                    )
                    for value in row
                )
                for row in covariance
            ),

        "source":
            f"{source_path}:{key_path}",
    }


def find_codebook_support():
    candidates = []

    for path in sorted(
        (S5 / "configs")
        .glob("*.json")
    ):
        try:
            obj = load_json(
                path
            )
        except Exception:
            continue

        def walk(
            value,
            prefix="",
        ):
            if not isinstance(
                value,
                dict,
            ):
                if isinstance(
                    value,
                    list,
                ):
                    for index, item in enumerate(
                        value
                    ):
                        walk(
                            item,
                            f"{prefix}[{index}]",
                        )

                return

            keys = {
                str(key).lower():
                    key
                for key in value
            }

            min_aliases = (
                "support_min_azimuth_rad",
                "azimuth_support_min_rad",
                "min_azimuth_rad",
            )

            max_aliases = (
                "support_max_azimuth_rad",
                "azimuth_support_max_rad",
                "max_azimuth_rad",
            )

            min_key = next(
                (
                    keys[
                        alias
                    ]
                    for alias in min_aliases
                    if alias in keys
                ),
                None,
            )

            max_key = next(
                (
                    keys[
                        alias
                    ]
                    for alias in max_aliases
                    if alias in keys
                ),
                None,
            )

            if (
                min_key is not None
                and
                max_key is not None
            ):
                lo = float(
                    value[
                        min_key
                    ]
                )

                hi = float(
                    value[
                        max_key
                    ]
                )

                if (
                    math.isfinite(
                        lo
                    )
                    and
                    math.isfinite(
                        hi
                    )
                    and
                    lo
                    <
                    hi
                ):
                    candidates.append(
                        (
                            str(
                                path
                            ),
                            prefix,
                            lo,
                            hi,
                        )
                    )

            for key, item in value.items():
                child = (
                    f"{prefix}.{key}"
                    if prefix
                    else str(key)
                )

                walk(
                    item,
                    child,
                )

        walk(
            obj
        )

    if not candidates:
        # Frozen module constants fallback.
        import iscai_stage5.beam_codebook as cb

        attrs = {
            name.lower():
                getattr(
                    cb,
                    name
                )
            for name in dir(
                cb
            )
            if not name.startswith(
                "_"
            )
        }

        lo = None
        hi = None

        for name, value in attrs.items():
            if (
                isinstance(
                    value,
                    (int, float),
                )
                and
                "azimuth" in name
                and
                "support" in name
            ):
                if "min" in name:
                    lo = float(
                        value
                    )

                if "max" in name:
                    hi = float(
                        value
                    )

        if (
            lo is not None
            and
            hi is not None
            and
            lo < hi
        ):
            return (
                lo,
                hi,
                "iscai_stage5.beam_codebook frozen constants",
            )

        raise RuntimeError(
            "Could not resolve frozen codebook "
            "azimuth support."
        )

    unique = sorted(
        set(
            (
                round(
                    item[
                        2
                    ],
                    15,
                ),
                round(
                    item[
                        3
                    ],
                    15,
                ),
            )
            for item in candidates
        )
    )

    if len(unique) != 1:
        raise RuntimeError(
            "Ambiguous frozen codebook support: "
            f"{candidates}"
        )

    lo, hi = unique[
        0
    ]

    return (
        float(
            lo
        ),
        float(
            hi
        ),
        (
            candidates[
                0
            ][
                0
            ]
            +
            ":"
            +
            candidates[
                0
            ][
                1
            ]
        ),
    )


def frozen_stage5_parameters():
    receiver_policy = load_json(
        RECEIVER_CONFIG
    )

    angular_policy = load_json(
        ANGULAR_CONFIG
    )

    latency_policy = load_json(
        LATENCY_CONFIG
    )

    minimum_forward = float(
        receiver_policy.get(
            "minimum_forward_H0_m",
            receiver_policy.get(
                "receiver_policy",
                {},
            ).get(
                "minimum_forward_H0_m",
                0.0,
            ),
        )
    )

    max_range = receiver_policy.get(
        "max_planar_range_m",
        receiver_policy.get(
            "receiver_policy",
            {},
        ).get(
            "max_planar_range_m",
            None,
        ),
    )

    if max_range is not None:
        max_range = float(
            max_range
        )

    sample_count = int(
        unique_numeric_key(
            angular_policy,
            "sample_count",
            default=2048,
        )
    )

    base_seed = int(
        unique_numeric_key(
            angular_policy,
            "base_seed",
            default=BASE_SEED,
        )
    )

    low_speed = float(
        unique_numeric_key(
            angular_policy,
            "low_speed_threshold_mps",
            default=0.25,
        )
    )

    multi_rate = latency_policy[
        "multi_rate_timing"
    ]

    tbeam = float(
        multi_rate[
            "Tbeam_s"
        ]
    )

    tframe = float(
        multi_rate[
            "Tframe_s"
        ]
    )

    if not (
        tbeam > 0.0
        and
        tframe > 0.0
        and
        tbeam < tframe
    ):
        raise RuntimeError(
            "Invalid frozen beam/frame timing."
        )

    offset = (
        find_receiver_offset()
    )

    support_min, support_max, support_source = (
        find_codebook_support()
    )

    return {
        "receiver_config":
            ReceiverSelectionConfig(
                minimum_forward_h0_m=minimum_forward,
                max_planar_range_m=max_range,
            ),

        "mc_sample_count":
            sample_count,

        "mc_base_seed":
            base_seed,

        "low_speed_threshold_mps":
            low_speed,

        "Tbeam_s":
            tbeam,

        "Tframe_s":
            tframe,

        "offset":
            offset,

        "support_min":
            support_min,

        "support_max":
            support_max,

        "support_source":
            support_source,
    }


# ============================================================
# Causal sample projection
# ============================================================

@dataclass
class ControllerSample:
    prediction_id: str
    semantic_class: str
    model_input: Any
    history: Any
    original_sample_index: int


def latest_history_heading(
    history,
    low_speed_threshold_mps: float,
) -> float:
    steps = tuple(
        history.steps
    )

    if not steps:
        return 0.0

    latest_index = int(
        history.latest_observed_frame_index
    )

    if not (
        0
        <=
        latest_index
        <
        len(
            steps
        )
    ):
        latest_index = (
            len(
                steps
            )
            -
            1
        )

    latest = steps[
        latest_index
    ]

    if bool(
        getattr(
            latest,
            "velocity_valid",
            False,
        )
    ):
        velocity = np.asarray(
            latest.velocity_H0_mps,
            dtype=np.float64,
        )

        speed = float(
            np.linalg.norm(
                velocity[
                    :2
                ]
            )
        )

        if (
            np.all(
                np.isfinite(
                    velocity
                )
            )
            and
            speed
            >=
            low_speed_threshold_mps
        ):
            return float(
                math.atan2(
                    velocity[
                        1
                    ],
                    velocity[
                        0
                    ],
                )
            )

    positions = []

    for step in steps[
        : latest_index + 1
    ]:
        if not bool(
            getattr(
                step,
                "observed",
                False,
            )
        ):
            continue

        position = np.asarray(
            step.position_H0_m,
            dtype=np.float64,
        )

        if (
            position.shape
            ==
            (
                3,
            )
            and
            np.all(
                np.isfinite(
                    position
                )
            )
        ):
            positions.append(
                position
            )

    if len(
        positions
    ) >= 2:
        delta = (
            positions[
                -1
            ][
                :2
            ]
            -
            positions[
                -2
            ][
                :2
            ]
        )

        speed_like = float(
            np.linalg.norm(
                delta
            )
        )

        if speed_like > 1e-8:
            return float(
                math.atan2(
                    delta[
                        1
                    ],
                    delta[
                        0
                    ],
                )
            )

    return 0.0


def project_causal_controller_samples(
    scene_inputs,
    samples,
    *,
    current_time_index: int,
):
    histories = tuple(
        scene_inputs.histories
    )

    history_by_id = {
        str(
            history.prediction_id
        ):
            history
        for history in histories
    }

    if len(
        history_by_id
    ) != len(
        histories
    ):
        raise RuntimeError(
            "Duplicate causal history prediction_id."
        )

    projected = []

    noncausal_mapping_count = 0
    unknown_class_count = 0

    for sample_index, sample in enumerate(
        samples
    ):
        prediction_id = str(
            sample.prediction_id
        )

        history = history_by_id.get(
            prediction_id
        )

        if history is None:
            raise RuntimeError(
                "Stage4 sample cannot bridge to "
                "causal history."
            )

        match_frame = int(
            sample.historical_match_frame_index
        )

        if match_frame > current_time_index:
            noncausal_mapping_count += 1
            continue

        try:
            semantic_class = str(
                normalize_semantic_class(
                    sample.actor_class
                )
            )

        except Exception:
            try:
                semantic_class = str(
                    normalize_semantic_class(
                        str(
                            sample.actor_class
                        )
                    )
                )

            except Exception:
                unknown_class_count += 1
                continue

        projected.append(
            ControllerSample(
                prediction_id=prediction_id,
                semantic_class=semantic_class,
                model_input=sample.model_input,
                history=history,
                original_sample_index=sample_index,
            )
        )

    projected.sort(
        key=lambda row:
            row.prediction_id
    )

    return {
        "rows":
            projected,

        "causal_history_count":
            len(
                histories
            ),

        "class_resolved_candidate_count":
            len(
                projected
            ),

        "unresolved_history_count":
            (
                len(
                    histories
                )
                -
                len(
                    projected
                )
            ),

        "noncausal_mapping_count":
            noncausal_mapping_count,

        "unknown_class_count":
            unknown_class_count,
    }


def choose_primary_receiver(
    projection,
    receiver_config,
):
    rows = projection[
        "rows"
    ]

    candidates = []

    for row in rows:
        position = tuple(
            float(
                value
            )
            for value in (
                row
                .history
                .latest_position_H0_m
            )
        )

        if (
            len(
                position
            )
            !=
            3
            or
            not all(
                math.isfinite(
                    value
                )
                for value in position
            )
        ):
            raise RuntimeError(
                "Invalid causal receiver-candidate position."
            )

        candidates.append(
            ReceiverCandidate(
                semantic_class=(
                    row.semantic_class
                ),
                position_h0_m=position,
                current_available=True,
                association_valid=True,
            )
        )

    result = (
        select_primary_receiver(
            candidates,
            receiver_config,
        )
    )

    selected_index = (
        result.selected_candidate_index
    )

    if selected_index is None:
        return (
            result,
            None,
        )

    selected_index = int(
        selected_index
    )

    if not (
        0
        <=
        selected_index
        <
        len(
            rows
        )
    ):
        raise RuntimeError(
            "Receiver selector returned invalid index."
        )

    return (
        result,
        rows[
            selected_index
        ],
    )


# ============================================================
# Exact Stage4 input tensorization
# ============================================================

def payload_to_dummy_supervision_arrays(
    payload,
):
    # Controller arrays contain causal model input only.
    # Future arrays are zero/invalid placeholders solely
    # because the frozen GaussianNormalizer.prepare API
    # expects the full training-array dictionary.
    # They are NEVER used in model.forward().
    target = np.asarray(
        payload.target_history,
        dtype=np.float32,
    )

    neighbors = np.asarray(
        payload.neighbor_histories,
        dtype=np.float32,
    )

    neighbor_mask = np.asarray(
        payload.neighbor_mask,
        dtype=np.bool_,
    )

    map_context = np.asarray(
        payload.map_context,
        dtype=np.float32,
    )

    if target.shape != (
        11,
        14,
    ):
        raise RuntimeError(
            f"Frozen target shape changed: "
            f"{target.shape}"
        )

    if neighbors.shape != (
        8,
        11,
        14,
    ):
        raise RuntimeError(
            f"Frozen neighbor shape changed: "
            f"{neighbors.shape}"
        )

    if neighbor_mask.shape != (
        8,
    ):
        raise RuntimeError(
            f"Frozen neighbor mask shape changed: "
            f"{neighbor_mask.shape}"
        )

    if map_context.shape != (
        10,
    ):
        raise RuntimeError(
            f"Frozen map-context shape changed: "
            f"{map_context.shape}"
        )

    return {
        "target":
            target[
                None,
                ...,
            ],

        "neighbors":
            neighbors[
                None,
                ...,
            ],

        "neighbor_mask":
            neighbor_mask[
                None,
                ...,
            ],

        "map_context":
            map_context[
                None,
                ...,
            ],

        "class_id":
            np.zeros(
                (
                    1,
                ),
                dtype=np.int64,
            ),

        "future":
            np.zeros(
                (
                    1,
                    4,
                    3,
                ),
                dtype=np.float32,
            ),

        "future_mask":
            np.zeros(
                (
                    1,
                    4,
                ),
                dtype=np.bool_,
            ),
    }


def calibrated_receiver_trajectory(
    runtime: FrozenGaussianRuntime,
    controller_row: ControllerSample,
):
    arrays = (
        payload_to_dummy_supervision_arrays(
            controller_row.model_input
        )
    )

    indices = np.asarray(
        [
            0
        ],
        dtype=np.int64,
    )

    batch = runtime.normalizer.prepare(
        arrays,
        indices,
        runtime.device,
    )

    torch.cuda.synchronize(
        runtime.device
    )

    started = time.perf_counter_ns()

    with torch.inference_mode():
        output = runtime.model(
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

    torch.cuda.synchronize(
        runtime.device
    )

    inference_runtime_s = (
        time.perf_counter_ns()
        -
        started
    ) / 1e9

    mean_norm = output.mean
    scale_norm = output.scale_tril

    if mean_norm.shape != (
        1,
        4,
        3,
    ):
        raise RuntimeError(
            "Gaussian mean output shape changed."
        )

    if scale_norm.shape != (
        1,
        4,
        3,
        3,
    ):
        raise RuntimeError(
            "Gaussian scale output shape changed."
        )

    variance_scale = torch.as_tensor(
        runtime.variance_scale,
        dtype=scale_norm.dtype,
        device=runtime.device,
    )

    scale_calibrated = (
        apply_variance_scale(
            scale_norm,
            variance_scale,
        )
    )

    label_mean = (
        runtime
        .normalizer
        .label_mean
        .to(
            runtime.device
        )
    )

    label_std = (
        runtime
        .normalizer
        .label_std
        .to(
            runtime.device
        )
    )

    if label_mean.shape != (
        4,
        3,
    ):
        raise RuntimeError(
            "Frozen label mean shape changed."
        )

    if label_std.shape != (
        4,
        3,
    ):
        raise RuntimeError(
            "Frozen label std shape changed."
        )

    displacement_mean_m = (
        label_mean[
            None,
            ...,
        ]
        +
        mean_norm
        *
        label_std[
            None,
            ...,
        ]
    )

    D = torch.diag_embed(
        label_std
    )

    scale_m = torch.matmul(
        D[
            None,
            ...,
        ],
        scale_calibrated,
    )

    covariance_m2 = torch.matmul(
        scale_m,
        scale_m.transpose(
            -1,
            -2,
        ),
    )

    current = torch.as_tensor(
        controller_row
        .history
        .latest_position_H0_m,
        dtype=mean_norm.dtype,
        device=runtime.device,
    )

    mean_absolute = (
        displacement_mean_m
        +
        current[
            None,
            None,
            :,
        ]
    )

    mean_np = (
        mean_absolute[
            0
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )

    covariance_np = (
        covariance_m2[
            0
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )

    if not (
        np.all(
            np.isfinite(
                mean_np
            )
        )
        and
        np.all(
            np.isfinite(
                covariance_np
            )
        )
    ):
        raise RuntimeError(
            "Non-finite calibrated Gaussian prediction."
        )

    for horizon in range(
        4
    ):
        eigenvalues = np.linalg.eigvalsh(
            0.5
            *
            (
                covariance_np[
                    horizon
                ]
                +
                covariance_np[
                    horizon
                ].T
            )
        )

        if np.min(
            eigenvalues
        ) <= 0.0:
            raise RuntimeError(
                "Calibrated trajectory covariance "
                "is not positive definite."
            )

    return (
        mean_np,
        covariance_np,
        float(
            inference_runtime_s
        ),
    )


# ============================================================
# Geometry / codebook helpers
# ============================================================

def offset_objects(
    parameters,
):
    offset = parameters[
        "offset"
    ]

    mean = offset[
        "mean_body_m"
    ]

    covariance = offset[
        "covariance_body_m2"
    ]

    geometry_offset = (
        ReceiverOffsetDistribution(
            mean_body_m=mean,
            covariance_body_m2=covariance,
        )
    )

    angular_offset = (
        ReceiverOffsetGaussianBody(
            mean_body_m=mean,
            covariance_body_m2=covariance,
        )
    )

    return (
        geometry_offset,
        angular_offset,
    )


def build_codebooks(
    parameters,
):
    result = {}

    for beam_count in (
        CODEBOOK_SIZES
    ):
        result[
            beam_count
        ] = (
            build_uniform_azimuth_codebook(
                beam_count=beam_count,
                support_min_azimuth_rad=(
                    parameters[
                        "support_min"
                    ]
                ),
                support_max_azimuth_rad=(
                    parameters[
                        "support_max"
                    ]
                ),
            )
        )

    return result


def receiver_current_azimuth(
    controller_row,
    mode,
    geometry_offset,
    current_heading,
):
    offset = (
        None
        if mode
        ==
        "centroid_baseline"
        else
        geometry_offset
    )

    distribution = (
        receiver_geometry_distribution_h0(
            actor_center_h0_m=(
                controller_row
                .history
                .latest_position_H0_m
            ),
            heading_h0_rad=current_heading,
            mode=mode,
            receiver_offset=offset,
        )
    )

    position = np.asarray(
        distribution.receiver_mean_h0_m,
        dtype=np.float64,
    )

    return float(
        math.atan2(
            position[
                1
            ],
            position[
                0
            ],
        )
    )


def selection_indices(
    value,
):
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
                int(
                    index
                )
                for index in item
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
                int(
                    item
                ),
            )

    raise RuntimeError(
        f"Could not extract beam indices from "
        f"{type(value).__name__}"
    )


def parse_temporal_result(
    result,
    previous_state,
):
    if isinstance(
        result,
        tuple,
    ):
        state = None
        decision = None

        for item in result:
            if isinstance(
                item,
                AdaptiveTemporalState,
            ):
                state = item

            if isinstance(
                item,
                AdaptiveTemporalDecision,
            ):
                decision = item

        if (
            state is not None
            and
            decision is not None
        ):
            return (
                state,
                decision,
            )

    if isinstance(
        result,
        AdaptiveTemporalDecision,
    ):
        decision = result

        state = AdaptiveTemporalState(
            previous_primary_beam_index=int(
                decision.primary_beam_index
            ),
            step_count=int(
                previous_state.step_count
            )
            +
            1,
            cumulative_switch_events=int(
                previous_state
                .cumulative_switch_events
            )
            +
            int(
                bool(
                    decision.primary_switched
                )
            ),
        )

        return (
            state,
            decision,
        )

    state = getattr(
        result,
        "state",
        None,
    )

    decision = getattr(
        result,
        "decision",
        None,
    )

    if (
        isinstance(
            state,
            AdaptiveTemporalState,
        )
        and
        isinstance(
            decision,
            AdaptiveTemporalDecision,
        )
    ):
        return (
            state,
            decision,
        )

    raise RuntimeError(
        "Could not parse adaptive_temporal_step "
        "return contract."
    )


def adaptive_probe_indices(
    decision: AdaptiveTemporalDecision,
):
    indices = set(
        int(
            value
        )
        for value in (
            decision
            .adaptive_selection
            .beam_indices
        )
    )

    indices.update(
        int(
            value
        )
        for value in (
            decision
            .local_neighbor_probe_indices
        )
    )

    widened = (
        decision
        .widened_fallback
    )

    if (
        bool(
            widened.available
        )
        and
        widened.fallback_beam_index
        is not None
    ):
        indices.add(
            int(
                widened
                .fallback_beam_index
            )
        )

    indices.update(
        int(
            value
        )
        for value in (
            decision
            .exhaustive_fallback_indices
        )
    )

    return tuple(
        sorted(
            indices
        )
    )


def circular_std(
    samples,
):
    values = np.asarray(
        samples,
        dtype=np.float64,
    )

    c = float(
        np.mean(
            np.cos(
                values
            )
        )
    )

    s = float(
        np.mean(
            np.sin(
                values
            )
        )
    )

    r = min(
        1.0,
        max(
            1e-15,
            math.hypot(
                c,
                s,
            ),
        ),
    )

    return float(
        math.sqrt(
            max(
                0.0,
                -2.0
                *
                math.log(
                    r
                ),
            )
        )
    )


def in_decision_cell(
    azimuth,
    cell,
    *,
    is_last,
):
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
    azimuth,
    indices,
    codebook,
):
    index_set = set(
        int(
            value
        )
        for value in indices
    )

    for cell in (
        codebook.cells
    ):
        if int(
            cell.index
        ) not in index_set:
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


def best_optical_link(
    *,
    indices,
    azimuth,
    elevation,
    codebook,
    probing_beam_count,
    Tbeam_s,
    Tframe_s,
):
    indices = tuple(
        int(
            value
        )
        for value in indices
    )

    if not indices:
        return None

    candidates = []

    for index in indices:
        if not (
            0
            <=
            index
            <
            len(
                codebook.cells
            )
        ):
            raise RuntimeError(
                "Beam index outside codebook."
            )

        cell = codebook.cells[
            index
        ]

        _, gain = (
            optical_gain_for_cell(
                receiver_azimuth_rad=azimuth,
                receiver_elevation_rad=elevation,
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
            item[
                0
            ],
            item[
                1
            ],
        )
    )

    cell = candidates[
        0
    ][
        2
    ]

    result = (
        evaluate_optical_link(
            receiver_azimuth_rad=azimuth,
            receiver_elevation_rad=elevation,
            cell=cell,
            probing_beam_count=int(
                probing_beam_count
            ),
            beam_probe_time_s=Tbeam_s,
            frame_time_s=Tframe_s,
        )
    )

    return result


def link_metrics(
    achieved,
    oracle,
):
    if (
        achieved is None
        or
        oracle is None
    ):
        return {
            "optical_gain":
                None,

            "received_power_normalized":
                None,

            "snr_db":
                None,

            "ber":
                None,

            "effective_rate_bps":
                None,

            "beam_gain_loss_db":
                None,

            "received_power_loss_db":
                None,

            "snr_loss_db":
                None,

            "effective_rate_loss_bps":
                None,
        }

    epsilon = 1e-300

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

        "ber":
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


# ============================================================
# Future truth — evaluator only
# ============================================================

def anchor_transform(
    built,
    current_index,
):
    matrix = np.asarray(
        built[
            "adapted"
        ]
        .frames
        .T_H0_from_W,
        dtype=np.float64,
    )

    if matrix.shape == (
        4,
        4,
    ):
        return matrix

    if (
        matrix.ndim
        ==
        3
        and
        matrix.shape[
            1:
        ]
        ==
        (
            4,
            4,
        )
    ):
        if not (
            0
            <=
            current_index
            <
            matrix.shape[
                0
            ]
        ):
            raise RuntimeError(
                "Anchor transform index invalid."
            )

        return matrix[
            current_index
        ]

    raise RuntimeError(
        f"Unexpected T_H0_from_W shape "
        f"{matrix.shape}"
    )


def state_center_h0(
    state,
    T_H0_from_W,
):
    world = np.asarray(
        [
            float(
                state.center_x
            ),
            float(
                state.center_y
            ),
            float(
                state.center_z
            ),
            1.0,
        ],
        dtype=np.float64,
    )

    result = (
        T_H0_from_W
        @
        world
    )

    return result[
        :3
    ]


def state_heading_h0(
    state,
    T_H0_from_W,
):
    yaw = math.atan2(
        float(
            T_H0_from_W[
                1,
                0
            ]
        ),
        float(
            T_H0_from_W[
                0,
                0
            ]
        ),
    )

    return wrap_angle(
        float(
            state.heading
        )
        +
        yaw
    )


def rotate_body_offset(
    offset_body,
    heading,
):
    x, y, z = (
        float(
            value
        )
        for value in offset_body
    )

    c = math.cos(
        heading
    )

    s = math.sin(
        heading
    )

    return np.asarray(
        [
            c * x - s * y,
            s * x + c * y,
            z,
        ],
        dtype=np.float64,
    )


def receiver_truth_position(
    center_h0,
    heading_h0,
    *,
    mode,
    known_offset,
    uncertain_realization,
):
    if mode == "centroid_baseline":
        offset = np.zeros(
            3,
            dtype=np.float64,
        )

    elif mode == "known_receiver_offset":
        offset = np.asarray(
            known_offset,
            dtype=np.float64,
        )

    elif mode == "uncertain_receiver_offset":
        offset = np.asarray(
            uncertain_realization,
            dtype=np.float64,
        )

    else:
        raise RuntimeError(
            f"Unknown receiver geometry mode {mode}"
        )

    return (
        np.asarray(
            center_h0,
            dtype=np.float64,
        )
        +
        rotate_body_offset(
            offset,
            heading_h0,
        )
    )


def spherical_angles(
    position,
):
    x, y, z = (
        float(
            value
        )
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

    return (
        radius,
        math.atan2(
            y,
            x,
        ),
        math.atan2(
            z,
            planar,
        ),
    )


# ============================================================
# One formal scenario
# ============================================================

def process_scenario(
    *,
    rank,
    formal,
    validation_by_id,
    clean_config,
    degraded_config,
    runtime,
    parameters,
    codebooks,
    run_fingerprint,
):
    scenario_id = str(
        formal[
            "scenario_id"
        ]
    )

    cache_path = (
        CACHE_DIR
        /
        (
            f"{rank:03d}_"
            f"{scenario_id}.json"
        )
    )

    if cache_path.is_file():
        cached = load_json(
            cache_path
        )

        if (
            cached.get(
                "status"
            )
            ==
            "COMPLETE"
            and
            cached.get(
                "run_fingerprint"
            )
            ==
            run_fingerprint
            and
            cached.get(
                "scenario_id"
            )
            ==
            scenario_id
        ):
            return (
                cached,
                True,
            )

        raise RuntimeError(
            "Existing Block5.8 scene cache has "
            "incompatible fingerprint. "
            f"Move it aside before retrying: {cache_path}"
        )

    validation_row = (
        validation_by_id.get(
            scenario_id
        )
    )

    if validation_row is None:
        raise RuntimeError(
            "Formal scenario absent from "
            "canonical validation manifest."
        )

    if (
        str(
            validation_row.source_shard
        )
        !=
        str(
            formal[
                "source_shard"
            ]
        )
    ):
        raise RuntimeError(
            "Formal source-shard mismatch."
        )

    if (
        str(
            validation_row.selection_hash
        )
        !=
        str(
            formal[
                "selection_hash"
            ]
        )
    ):
        raise RuntimeError(
            "Formal selection-hash mismatch."
        )

    scenario = read_motion_scenario(
        validation_row,
        paired_root=PAIRED_ROOT,
        compact_record_offset=int(
            formal[
                "compact_record_offset"
            ]
        ),
    )

    if (
        str(
            scenario.scenario_id
        )
        !=
        scenario_id
    ):
        raise RuntimeError(
            "Physical scenario-ID mismatch."
        )

    current_index = int(
        scenario.current_time_index
    )

    if current_index != 10:
        raise RuntimeError(
            "Frozen current_time_index changed."
        )

    built = build_real_causal_inputs(
        scenario,
        clean_config=clean_config,
        degraded_config=degraded_config,
    )

    scene_inputs = built[
        "scene_inputs"
    ]

    # This helper is used ONLY to recover the frozen
    # prediction_id -> ModelInputPayload / semantic-class
    # bridge already audited in Route-B.
    #
    # future_label itself is never read by controller code.
    samples = attach_supervision(
        scene_inputs,
        scenario,
        T_H0_from_W=(
            built[
                "adapted"
            ]
            .frames
            .T_H0_from_W
        ),
    )

    projection = (
        project_causal_controller_samples(
            scene_inputs,
            samples,
            current_time_index=current_index,
        )
    )

    if (
        projection[
            "noncausal_mapping_count"
        ]
        !=
        0
    ):
        raise RuntimeError(
            "Non-causal sample mapping detected."
        )

    receiver_result, selected = (
        choose_primary_receiver(
            projection,
            parameters[
                "receiver_config"
            ],
        )
    )

    base_result = {
        "stage":
            5,

        "block":
            "5.8",

        "status":
            "COMPLETE",

        "rank":
            int(
                rank
            ),

        "scenario_id":
            scenario_id,

        "run_fingerprint":
            run_fingerprint,

        "causal_population": {
            "causal_history_count":
                int(
                    projection[
                        "causal_history_count"
                    ]
                ),

            "class_resolved_candidate_count":
                int(
                    projection[
                        "class_resolved_candidate_count"
                    ]
                ),

            "unresolved_history_count":
                int(
                    projection[
                        "unresolved_history_count"
                    ]
                ),

            "unknown_class_count":
                int(
                    projection[
                        "unknown_class_count"
                    ]
                ),
        },

        "receiver": {
            "status":
                str(
                    receiver_result.status
                ),

            "eligible_candidate_count":
                int(
                    receiver_result
                    .eligible_candidate_count
                ),

            "selected":
                bool(
                    selected
                    is not None
                ),

            "prediction_id":
                (
                    selected.prediction_id
                    if selected
                    is not None
                    else
                    None
                ),

            "semantic_class":
                (
                    selected.semantic_class
                    if selected
                    is not None
                    else
                    None
                ),

            "planar_range_m":
                (
                    float(
                        receiver_result
                        .planar_range_m
                    )
                    if
                    receiver_result
                    .planar_range_m
                    is not None
                    else
                    None
                ),
        },

        "availability": {
            "eligible_communication_receiver":
                bool(
                    selected
                    is not None
                ),

            "selected_receiver_prediction_available":
                bool(
                    selected
                    is not None
                ),

            "valid_future_receiver_endpoint":
                {
                    str(
                        horizon
                    ):
                        False
                    for horizon in (
                        HORIZONS_S
                    )
                },
        },

        "adaptive_records":
            [],

        "baseline_records":
            [],

        "runtime": {
            "stage4_predictor_inference_runtime_s":
                None,
        },

        "scientific_execution": {
            "tracks_to_predict_receiver_selection":
                False,

            "future_truth_receiver_selection":
                False,

            "future_label_read_by_controller":
                False,

            "future_truth_access_after_controller_decisions":
                False,

            "training":
                False,

            "recalibration":
                False,

            "model_selection":
                False,

            "threshold_tuning":
                False,
        },
    }

    if selected is None:
        atomic_json(
            cache_path,
            base_result,
        )

        return (
            base_result,
            False,
        )

    current_heading = (
        latest_history_heading(
            selected.history,
            parameters[
                "low_speed_threshold_mps"
            ],
        )
    )

    trajectory_mean, trajectory_covariance, inference_runtime = (
        calibrated_receiver_trajectory(
            runtime,
            selected,
        )
    )

    base_result[
        "runtime"
    ][
        "stage4_predictor_inference_runtime_s"
    ] = (
        inference_runtime
    )

    geometry_offset, angular_offset = (
        offset_objects(
            parameters
        )
    )

    current_position = tuple(
        float(
            value
        )
        for value in (
            selected
            .history
            .latest_position_H0_m
        )
    )

    # ========================================================
    # PRE-TRUTH CONTROLLER DECISION PHASE
    # ========================================================

    plans = []

    for mode in (
        GEOMETRY_MODES
    ):
        receiver_offset = (
            None
            if mode
            ==
            "centroid_baseline"
            else
            angular_offset
        )

        torch.cuda.synchronize(
            runtime.device
        )

        started = (
            time.perf_counter_ns()
        )

        posterior = (
            deterministic_receiver_angular_posterior(
                trajectory_mean_h0_m=(
                    trajectory_mean
                ),
                raw_trajectory_covariance_h0_m2=(
                    trajectory_covariance
                ),
                current_position_h0_m=(
                    current_position
                ),
                current_heading_h0_rad=(
                    current_heading
                ),
                receiver_geometry_mode=mode,
                receiver_offset=receiver_offset,
                sample_count=int(
                    parameters[
                        "mc_sample_count"
                    ]
                ),
                seed=stable_seed(
                    parameters[
                        "mc_base_seed"
                    ],
                    scenario_id,
                    mode,
                    "posterior",
                ),
                low_speed_threshold_mps=float(
                    parameters[
                        "low_speed_threshold_mps"
                    ]
                ),
            )
        )

        posterior_runtime_s = (
            time.perf_counter_ns()
            -
            started
        ) / 1e9

        azimuth_samples = np.asarray(
            posterior.azimuth_samples_rad,
            dtype=np.float64,
        )

        if azimuth_samples.shape != (
            parameters[
                "mc_sample_count"
            ],
            4,
        ):
            raise RuntimeError(
                "Angular MC azimuth sample shape changed."
            )

        current_azimuth = (
            receiver_current_azimuth(
                selected,
                mode,
                geometry_offset,
                current_heading,
            )
        )

        for beam_count in (
            CODEBOOK_SIZES
        ):
            codebook = codebooks[
                beam_count
            ]

            temporal_states = {
                q:
                    initial_adaptive_temporal_state()
                for q in REQUESTED_Q
            }

            geometry_selection = (
                geometry_nearest_beam(
                    predicted_azimuth_rad=(
                        current_azimuth
                    ),
                    codebook=codebook,
                )
            )

            geometry_indices = (
                selection_indices(
                    geometry_selection
                )
            )

            persistence_indices = (
                geometry_indices
            )

            for horizon_index, horizon_s in enumerate(
                HORIZONS_S
            ):
                beam_started = (
                    time.perf_counter_ns()
                )

                probability = (
                    beam_probability_mass_from_samples(
                        azimuth_samples_rad=(
                            azimuth_samples[
                                :,
                                horizon_index
                            ]
                        ),
                        codebook=codebook,
                    )
                )

                beam_probability_runtime_s = (
                    time.perf_counter_ns()
                    -
                    beam_started
                ) / 1e9

                fixed = {}

                for k in FIXED_TOPK:
                    fixed[
                        k
                    ] = (
                        fixed_top_k_probability(
                            probability=probability,
                            codebook=codebook,
                            k=k,
                        )
                    )

                adaptive = {}

                for q in REQUESTED_Q:
                    selection_started = (
                        time.perf_counter_ns()
                    )

                    result = (
                        adaptive_temporal_step(
                            state=(
                                temporal_states[
                                    q
                                ]
                            ),
                            probability=probability,
                            codebook=codebook,
                            requested_coverage=q,
                        )
                    )

                    selection_runtime_s = (
                        time.perf_counter_ns()
                        -
                        selection_started
                    ) / 1e9

                    next_state, decision = (
                        parse_temporal_result(
                            result,
                            temporal_states[
                                q
                            ],
                        )
                    )

                    temporal_states[
                        q
                    ] = next_state

                    adaptive[
                        q
                    ] = {
                        "decision":
                            decision,

                        "selection_runtime_s":
                            float(
                                selection_runtime_s
                            ),
                    }

                plans.append(
                    {
                        "geometry_mode":
                            mode,

                        "beam_count":
                            beam_count,

                        "horizon_index":
                            horizon_index,

                        "horizon_s":
                            horizon_s,

                        "posterior":
                            posterior,

                        "posterior_mapping_runtime_s":
                            float(
                                posterior_runtime_s
                            ),

                        "beam_probability_runtime_s":
                            float(
                                beam_probability_runtime_s
                            ),

                        "probability":
                            probability,

                        "fixed":
                            fixed,

                        "geometry_indices":
                            geometry_indices,

                        "persistence_indices":
                            persistence_indices,

                        "adaptive":
                            adaptive,

                        "azimuth_std_rad":
                            circular_std(
                                azimuth_samples[
                                    :,
                                    horizon_index
                                ]
                            ),
                    }
                )

    # ========================================================
    # ONLY NOW: FUTURE TRUTH / EVALUATOR PHASE
    # ========================================================

    base_result[
        "scientific_execution"
    ][
        "future_truth_access_after_controller_decisions"
    ] = True

    original_sample = samples[
        selected
        .original_sample_index
    ]

    truth_resolution = (
        resolve_sample_truth_track_index(
            original_sample
        )
    )

    truth_index = int(
        truth_resolution[
            "index"
        ]
    )

    if not (
        0
        <=
        truth_index
        <
        len(
            scenario.tracks
        )
    ):
        raise RuntimeError(
            "Evaluator truth-track index invalid."
        )

    truth_track = scenario.tracks[
        truth_index
    ]

    T_H0_from_W = (
        anchor_transform(
            built,
            current_index,
        )
    )

    mean_offset = np.asarray(
        parameters[
            "offset"
        ][
            "mean_body_m"
        ],
        dtype=np.float64,
    )

    covariance_offset = np.asarray(
        parameters[
            "offset"
        ][
            "covariance_body_m2"
        ],
        dtype=np.float64,
    )

    truth_rng = np.random.default_rng(
        stable_seed(
            scenario_id,
            "formal_receiver_offset_truth",
        )
    )

    uncertain_realization = (
        truth_rng.multivariate_normal(
            mean_offset,
            covariance_offset,
            check_valid="raise",
        )
    )

    truth_by_mode = {
        mode:
            {}
        for mode in (
            GEOMETRY_MODES
        )
    }

    for horizon_index, (
        horizon_s,
        frame_offset,
    ) in enumerate(
        zip(
            HORIZONS_S,
            HORIZON_OFFSETS,
        )
    ):
        state_index = (
            current_index
            +
            frame_offset
        )

        valid = (
            state_index
            <
            len(
                truth_track.states
            )
            and
            bool(
                truth_track.states[
                    state_index
                ].valid
            )
        )

        base_result[
            "availability"
        ][
            "valid_future_receiver_endpoint"
        ][
            str(
                horizon_s
            )
        ] = bool(
            valid
        )

        if not valid:
            continue

        state = truth_track.states[
            state_index
        ]

        center = state_center_h0(
            state,
            T_H0_from_W,
        )

        heading = state_heading_h0(
            state,
            T_H0_from_W,
        )

        for mode in (
            GEOMETRY_MODES
        ):
            position = (
                receiver_truth_position(
                    center,
                    heading,
                    mode=mode,
                    known_offset=(
                        mean_offset
                    ),
                    uncertain_realization=(
                        uncertain_realization
                    ),
                )
            )

            radius, azimuth, elevation = (
                spherical_angles(
                    position
                )
            )

            truth_by_mode[
                mode
            ][
                horizon_index
            ] = {
                "range_m":
                    float(
                        radius
                    ),

                "azimuth_rad":
                    float(
                        azimuth
                    ),

                "elevation_rad":
                    float(
                        elevation
                    ),
            }

    Tbeam_s = float(
        parameters[
            "Tbeam_s"
        ]
    )

    Tframe_s = float(
        parameters[
            "Tframe_s"
        ]
    )

    for plan in plans:
        mode = plan[
            "geometry_mode"
        ]

        horizon_index = plan[
            "horizon_index"
        ]

        horizon_s = plan[
            "horizon_s"
        ]

        beam_count = plan[
            "beam_count"
        ]

        truth = (
            truth_by_mode[
                mode
            ].get(
                horizon_index
            )
        )

        if truth is None:
            continue

        codebook = codebooks[
            beam_count
        ]

        truth_azimuth = float(
            truth[
                "azimuth_rad"
            ]
        )

        truth_elevation = float(
            truth[
                "elevation_rad"
            ]
        )

        oracle_selection = (
            oracle_best_gain_beam(
                realized_azimuth_rad=(
                    truth_azimuth
                ),
                codebook=codebook,
            )
        )

        oracle_indices = (
            selection_indices(
                oracle_selection
            )
        )

        oracle_link = best_optical_link(
            indices=oracle_indices,
            azimuth=truth_azimuth,
            elevation=truth_elevation,
            codebook=codebook,
            probing_beam_count=1,
            Tbeam_s=Tbeam_s,
            Tframe_s=Tframe_s,
        )

        exhaustive_indices = tuple(
            range(
                beam_count
            )
        )

        baseline_sets = {
            "geometry_nearest":
                (
                    plan[
                        "geometry_indices"
                    ],
                    1,
                ),

            "previous_beam_persistence":
                (
                    plan[
                        "persistence_indices"
                    ],
                    1,
                ),

            "exhaustive":
                (
                    exhaustive_indices,
                    beam_count,
                ),

            "oracle":
                (
                    oracle_indices,
                    1,
                ),
        }

        for k in FIXED_TOPK:
            baseline_sets[
                f"fixed_top_{k}"
            ] = (
                selection_indices(
                    plan[
                        "fixed"
                    ][
                        k
                    ]
                ),
                k,
            )

        for policy, (
            indices,
            probe_count,
        ) in baseline_sets.items():
            link = best_optical_link(
                indices=indices,
                azimuth=truth_azimuth,
                elevation=truth_elevation,
                codebook=codebook,
                probing_beam_count=probe_count,
                Tbeam_s=Tbeam_s,
                Tframe_s=Tframe_s,
            )

            hit = (
                True
                if policy
                ==
                "oracle"
                else
                selected_set_hit(
                    truth_azimuth,
                    indices,
                    codebook,
                )
            )

            metrics = (
                link_metrics(
                    link,
                    oracle_link,
                )
            )

            base_result[
                "baseline_records"
            ].append(
                {
                    "scenario_id":
                        scenario_id,

                    "geometry_mode":
                        mode,

                    "codebook_size":
                        beam_count,

                    "horizon_s":
                        horizon_s,

                    "policy":
                        policy,

                    "hit":
                        bool(
                            hit
                        ),

                    "probe_count":
                        int(
                            probe_count
                        ),

                    "probing_overhead_fraction":
                        float(
                            probe_count
                            *
                            Tbeam_s
                            /
                            Tframe_s
                        ),

                    "overhead_reduction_vs_exhaustive":
                        float(
                            1.0
                            -
                            probe_count
                            /
                            beam_count
                        ),

                    "receiver_distance_m":
                        float(
                            truth[
                                "range_m"
                            ]
                        ),

                    "azimuth_uncertainty_std_rad":
                        float(
                            plan[
                                "azimuth_std_rad"
                            ]
                        ),

                    **metrics,
                }
            )

        for q in REQUESTED_Q:
            decision = (
                plan[
                    "adaptive"
                ][
                    q
                ][
                    "decision"
                ]
            )

            adaptive_indices = tuple(
                int(
                    value
                )
                for value in (
                    decision
                    .adaptive_selection
                    .beam_indices
                )
            )

            physical_indices = (
                adaptive_probe_indices(
                    decision
                )
            )

            physical_probe_count = int(
                actual_probe_count_for_adaptive_decision(
                    decision
                )
            )

            if physical_probe_count < len(
                physical_indices
            ):
                raise RuntimeError(
                    "Adaptive physical probe count "
                    "smaller than unique probe set."
                )

            hit = selected_set_hit(
                truth_azimuth,
                adaptive_indices,
                codebook,
            )

            link = best_optical_link(
                indices=physical_indices,
                azimuth=truth_azimuth,
                elevation=truth_elevation,
                codebook=codebook,
                probing_beam_count=(
                    physical_probe_count
                ),
                Tbeam_s=Tbeam_s,
                Tframe_s=Tframe_s,
            )

            metrics = link_metrics(
                link,
                oracle_link,
            )

            selected_mass = float(
                decision
                .adaptive_selection
                .selected_in_support_mass
            )

            adaptive_k = int(
                decision
                .adaptive_selection
                .k
            )

            extra_reacquisition_probes = (
                max(
                    0,
                    physical_probe_count
                    -
                    adaptive_k,
                )
                if bool(
                    decision.loss_of_lock
                )
                else
                0
            )

            base_result[
                "adaptive_records"
            ].append(
                {
                    "scenario_id":
                        scenario_id,

                    "geometry_mode":
                        mode,

                    "codebook_size":
                        beam_count,

                    "horizon_s":
                        horizon_s,

                    "requested_probability_mass":
                        float(
                            q
                        ),

                    "posterior_mass_selected":
                        selected_mass,

                    "selected_k":
                        adaptive_k,

                    "physical_probe_count":
                        physical_probe_count,

                    "probing_overhead_fraction":
                        float(
                            physical_probe_count
                            *
                            Tbeam_s
                            /
                            Tframe_s
                        ),

                    "overhead_reduction_vs_exhaustive":
                        float(
                            1.0
                            -
                            physical_probe_count
                            /
                            beam_count
                        ),

                    "containment_hit":
                        bool(
                            hit
                        ),

                    "beam_outage":
                        bool(
                            not hit
                        ),

                    "primary_beam_index":
                        int(
                            decision
                            .primary_beam_index
                        ),

                    "primary_switched":
                        bool(
                            decision
                            .primary_switched
                        ),

                    "loss_of_lock":
                        bool(
                            decision
                            .loss_of_lock
                        ),

                    "exhaustive_reacquisition":
                        bool(
                            decision
                            .exhaustive_fallback_active
                        ),

                    "reacquisition_latency_s":
                        float(
                            extra_reacquisition_probes
                            *
                            Tbeam_s
                        ),

                    "receiver_distance_m":
                        float(
                            truth[
                                "range_m"
                            ]
                        ),

                    "azimuth_uncertainty_std_rad":
                        float(
                            plan[
                                "azimuth_std_rad"
                            ]
                        ),

                    "posterior_mapping_runtime_s":
                        float(
                            plan[
                                "posterior_mapping_runtime_s"
                            ]
                        ),

                    "beam_probability_runtime_s":
                        float(
                            plan[
                                "beam_probability_runtime_s"
                            ]
                        ),

                    "topk_selection_runtime_s":
                        float(
                            plan[
                                "adaptive"
                            ][
                                q
                            ][
                                "selection_runtime_s"
                            ]
                        ),

                    **metrics,
                }
            )

    atomic_json(
        cache_path,
        base_result,
    )

    return (
        base_result,
        False,
    )


# ============================================================
# Aggregation
# ============================================================

def aggregate_adaptive(
    records,
):
    grouped = defaultdict(
        list
    )

    for row in records:
        key = (
            row[
                "geometry_mode"
            ],
            int(
                row[
                    "codebook_size"
                ]
            ),
            float(
                row[
                    "horizon_s"
                ]
            ),
            float(
                row[
                    "requested_probability_mass"
                ]
            ),
        )

        grouped[
            key
        ].append(
            row
        )

    result = {}

    for key, rows in sorted(
        grouped.items()
    ):
        mode, beam_count, horizon_s, q = key

        hits = int(
            sum(
                bool(
                    row[
                        "containment_hit"
                    ]
                )
                for row in rows
            )
        )

        trials = len(
            rows
        )

        coverage = (
            empirical_coverage_decision(
                hits=hits,
                trials=trials,
                requested_q=q,
            )
            .to_dict()
            if trials
            else
            {
                "status":
                    "NOT_EVALUABLE",

                "passed":
                    False,

                "hits":
                    0,

                "trials":
                    0,
            }
        )

        identifier = (
            f"{mode}|"
            f"{beam_count}|"
            f"{horizon_s}|"
            f"{q}"
        )

        result[
            identifier
        ] = {
            "geometry_mode":
                mode,

            "codebook_size":
                beam_count,

            "horizon_s":
                horizon_s,

            "requested_probability_mass":
                q,

            "trials":
                trials,

            "hits":
                hits,

            "empirical_coverage":
                (
                    float(
                        hits
                        /
                        trials
                    )
                    if trials
                    else
                    None
                ),

            "formal_coverage_decision":
                coverage,

            "average_selected_k":
                mean_or_none(
                    row[
                        "selected_k"
                    ]
                    for row in rows
                ),

            "average_physical_probe_count":
                mean_or_none(
                    row[
                        "physical_probe_count"
                    ]
                    for row in rows
                ),

            "average_posterior_mass_selected":
                mean_or_none(
                    row[
                        "posterior_mass_selected"
                    ]
                    for row in rows
                ),

            "average_probing_overhead_fraction":
                mean_or_none(
                    row[
                        "probing_overhead_fraction"
                    ]
                    for row in rows
                ),

            "average_overhead_reduction_vs_exhaustive":
                mean_or_none(
                    row[
                        "overhead_reduction_vs_exhaustive"
                    ]
                    for row in rows
                ),

            "beam_gain_loss_db":
                mean_or_none(
                    row[
                        "beam_gain_loss_db"
                    ]
                    for row in rows
                ),

            "received_power_loss_db":
                mean_or_none(
                    row[
                        "received_power_loss_db"
                    ]
                    for row in rows
                ),

            "snr_loss_db":
                mean_or_none(
                    row[
                        "snr_loss_db"
                    ]
                    for row in rows
                ),

            "mean_BER":
                mean_or_none(
                    row[
                        "ber"
                    ]
                    for row in rows
                ),

            "mean_effective_rate_bps":
                mean_or_none(
                    row[
                        "effective_rate_bps"
                    ]
                    for row in rows
                ),

            "effective_rate_loss_bps":
                mean_or_none(
                    row[
                        "effective_rate_loss_bps"
                    ]
                    for row in rows
                ),

            "outage_probability":
                mean_or_none(
                    1.0
                    if row[
                        "beam_outage"
                    ]
                    else
                    0.0
                    for row in rows
                ),

            "beam_switching_rate":
                mean_or_none(
                    1.0
                    if row[
                        "primary_switched"
                    ]
                    else
                    0.0
                    for row in rows
                ),

            "mean_reacquisition_latency_s":
                mean_or_none(
                    row[
                        "reacquisition_latency_s"
                    ]
                    for row in rows
                ),

            "latency": {
                "posterior_mapping_runtime_s":
                    mean_or_none(
                        row[
                            "posterior_mapping_runtime_s"
                        ]
                        for row in rows
                    ),

                "beam_probability_runtime_s":
                    mean_or_none(
                        row[
                            "beam_probability_runtime_s"
                        ]
                        for row in rows
                    ),

                "topk_selection_runtime_s":
                    mean_or_none(
                        row[
                            "topk_selection_runtime_s"
                        ]
                        for row in rows
                    ),
            },
        }

    return result


def aggregate_baselines(
    records,
):
    grouped = defaultdict(
        list
    )

    for row in records:
        key = (
            row[
                "policy"
            ],
            row[
                "geometry_mode"
            ],
            int(
                row[
                    "codebook_size"
                ]
            ),
            float(
                row[
                    "horizon_s"
                ]
            ),
        )

        grouped[
            key
        ].append(
            row
        )

    result = {}

    for key, rows in sorted(
        grouped.items()
    ):
        policy, mode, beam_count, horizon_s = key

        trials = len(
            rows
        )

        hits = int(
            sum(
                bool(
                    row[
                        "hit"
                    ]
                )
                for row in rows
            )
        )

        identifier = (
            f"{policy}|"
            f"{mode}|"
            f"{beam_count}|"
            f"{horizon_s}"
        )

        result[
            identifier
        ] = {
            "policy":
                policy,

            "geometry_mode":
                mode,

            "codebook_size":
                beam_count,

            "horizon_s":
                horizon_s,

            "trials":
                trials,

            "hits":
                hits,

            "hit_rate":
                (
                    float(
                        hits
                        /
                        trials
                    )
                    if trials
                    else
                    None
                ),

            "mean_probe_count":
                mean_or_none(
                    row[
                        "probe_count"
                    ]
                    for row in rows
                ),

            "mean_probing_overhead_fraction":
                mean_or_none(
                    row[
                        "probing_overhead_fraction"
                    ]
                    for row in rows
                ),

            "beam_gain_loss_db":
                mean_or_none(
                    row[
                        "beam_gain_loss_db"
                    ]
                    for row in rows
                ),

            "received_power_loss_db":
                mean_or_none(
                    row[
                        "received_power_loss_db"
                    ]
                    for row in rows
                ),

            "snr_loss_db":
                mean_or_none(
                    row[
                        "snr_loss_db"
                    ]
                    for row in rows
                ),

            "mean_BER":
                mean_or_none(
                    row[
                        "ber"
                    ]
                    for row in rows
                ),

            "mean_effective_rate_bps":
                mean_or_none(
                    row[
                        "effective_rate_bps"
                    ]
                    for row in rows
                ),

            "effective_rate_loss_bps":
                mean_or_none(
                    row[
                        "effective_rate_loss_bps"
                    ]
                    for row in rows
                ),
        }

    return result


def find_adaptive_stratum(
    aggregate,
    mode,
    beam_count,
    horizon_s,
    q,
):
    key = (
        f"{mode}|"
        f"{beam_count}|"
        f"{horizon_s}|"
        f"{q}"
    )

    return aggregate.get(
        key
    )


def find_baseline_stratum(
    aggregate,
    policy,
    mode,
    beam_count,
    horizon_s,
):
    key = (
        f"{policy}|"
        f"{mode}|"
        f"{beam_count}|"
        f"{horizon_s}"
    )

    return aggregate.get(
        key
    )


def formal_acceptance(
    adaptive_aggregate,
    baseline_aggregate,
):
    coverage_tests = []

    coverage_pass = True

    for beam_count in (
        CODEBOOK_SIZES
    ):
        for horizon_s in (
            HORIZONS_S
        ):
            stratum = (
                find_adaptive_stratum(
                    adaptive_aggregate,
                    PRIMARY_FORMAL_GEOMETRY,
                    beam_count,
                    horizon_s,
                    NOMINAL_Q,
                )
            )

            if (
                stratum is None
                or
                int(
                    stratum[
                        "trials"
                    ]
                )
                ==
                0
            ):
                decision = {
                    "status":
                        "NOT_EVALUABLE",

                    "passed":
                        False,
                }

                coverage_pass = False

            else:
                decision = (
                    stratum[
                        "formal_coverage_decision"
                    ]
                )

                if not bool(
                    decision[
                        "passed"
                    ]
                ):
                    coverage_pass = False

            coverage_tests.append(
                {
                    "geometry_mode":
                        PRIMARY_FORMAL_GEOMETRY,

                    "codebook_size":
                        beam_count,

                    "horizon_s":
                        horizon_s,

                    "requested_q":
                        NOMINAL_Q,

                    "decision":
                        decision,
                }
            )

    overhead_tests = []

    overhead_pass = True

    for beam_count in (
        CODEBOOK_SIZES
    ):
        adaptive_rows = []

        for horizon_s in (
            HORIZONS_S
        ):
            stratum = (
                find_adaptive_stratum(
                    adaptive_aggregate,
                    PRIMARY_FORMAL_GEOMETRY,
                    beam_count,
                    horizon_s,
                    NOMINAL_Q,
                )
            )

            if (
                stratum is not None
                and
                stratum[
                    "average_physical_probe_count"
                ]
                is not None
            ):
                adaptive_rows.append(
                    stratum
                )

        adaptive_mean_probes = (
            mean_or_none(
                row[
                    "average_physical_probe_count"
                ]
                for row in adaptive_rows
            )
        )

        hard_exhaustive_pass = (
            adaptive_mean_probes
            is not None
            and
            adaptive_mean_probes
            <
            beam_count
        )

        valid_fixed = []

        fixed_details = []

        for k in (
            FIXED_TOPK
        ):
            per_horizon = []

            all_horizons_pass = True

            for horizon_s in (
                HORIZONS_S
            ):
                fixed = (
                    find_baseline_stratum(
                        baseline_aggregate,
                        f"fixed_top_{k}",
                        PRIMARY_FORMAL_GEOMETRY,
                        beam_count,
                        horizon_s,
                    )
                )

                if (
                    fixed is None
                    or
                    int(
                        fixed[
                            "trials"
                        ]
                    )
                    ==
                    0
                ):
                    all_horizons_pass = False

                    per_horizon.append(
                        {
                            "horizon_s":
                                horizon_s,

                            "status":
                                "NOT_EVALUABLE",
                        }
                    )

                    continue

                decision = (
                    empirical_coverage_decision(
                        hits=int(
                            fixed[
                                "hits"
                            ]
                        ),
                        trials=int(
                            fixed[
                                "trials"
                            ]
                        ),
                        requested_q=NOMINAL_Q,
                    )
                )

                if not decision.passed:
                    all_horizons_pass = False

                per_horizon.append(
                    {
                        "horizon_s":
                            horizon_s,

                        "decision":
                            decision.to_dict(),
                    }
                )

            if all_horizons_pass:
                valid_fixed.append(
                    k
                )

            fixed_details.append(
                {
                    "k":
                        k,

                    "coverage_valid":
                        all_horizons_pass,

                    "horizons":
                        per_horizon,
                }
            )

        best_valid_fixed = (
            min(
                valid_fixed
            )
            if valid_fixed
            else
            None
        )

        if best_valid_fixed is None:
            fixed_comparison_pass = None

        else:
            fixed_comparison_pass = bool(
                adaptive_mean_probes
                is not None
                and
                adaptive_mean_probes
                <=
                best_valid_fixed
                +
                1e-12
            )

        codebook_pass = (
            bool(
                hard_exhaustive_pass
            )
            and
            (
                fixed_comparison_pass
                is not False
            )
        )

        if not codebook_pass:
            overhead_pass = False

        overhead_tests.append(
            {
                "codebook_size":
                    beam_count,

                "adaptive_mean_physical_probe_count":
                    adaptive_mean_probes,

                "hard_exhaustive_requirement_pass":
                    bool(
                        hard_exhaustive_pass
                    ),

                "coverage_valid_fixed_topk":
                    valid_fixed,

                "best_coverage_valid_fixed_topk":
                    best_valid_fixed,

                "adaptive_vs_valid_fixed_requirement_pass":
                    fixed_comparison_pass,

                "fixed_details":
                    fixed_details,

                "codebook_overhead_pass":
                    codebook_pass,
            }
        )

    return {
        "primary_geometry_mode":
            PRIMARY_FORMAL_GEOMETRY,

        "coverage_acceptance_pass":
            bool(
                coverage_pass
            ),

        "coverage_tests":
            coverage_tests,

        "overhead_reduction_acceptance_pass":
            bool(
                overhead_pass
            ),

        "overhead_tests":
            overhead_tests,

        "stage5_pdf_acceptance_pass":
            bool(
                coverage_pass
                and
                overhead_pass
            ),
    }


def reliability_overhead_curve(
    adaptive_aggregate,
):
    result = {}

    for mode in (
        GEOMETRY_MODES
    ):
        for beam_count in (
            CODEBOOK_SIZES
        ):
            points = []

            for q in (
                REQUESTED_Q
            ):
                strata = [
                    value
                    for value in (
                        adaptive_aggregate
                        .values()
                    )
                    if (
                        value[
                            "geometry_mode"
                        ]
                        ==
                        mode
                        and
                        int(
                            value[
                                "codebook_size"
                            ]
                        )
                        ==
                        beam_count
                        and
                        float(
                            value[
                                "requested_probability_mass"
                            ]
                        )
                        ==
                        q
                    )
                ]

                points.append(
                    {
                        "requested_q":
                            q,

                        "empirical_coverage":
                            mean_or_none(
                                row[
                                    "empirical_coverage"
                                ]
                                for row in strata
                            ),

                        "mean_physical_probe_count":
                            mean_or_none(
                                row[
                                    "average_physical_probe_count"
                                ]
                                for row in strata
                            ),

                        "mean_overhead_fraction":
                            mean_or_none(
                                row[
                                    "average_probing_overhead_fraction"
                                ]
                                for row in strata
                            ),
                    }
                )

            valid_points = [
                point
                for point in points
                if (
                    point[
                        "empirical_coverage"
                    ]
                    is not None
                    and
                    point[
                        "mean_physical_probe_count"
                    ]
                    is not None
                )
            ]

            pareto = []

            for point in valid_points:
                dominated = False

                for other in (
                    valid_points
                ):
                    if other is point:
                        continue

                    better_or_equal = (
                        other[
                            "empirical_coverage"
                        ]
                        >=
                        point[
                            "empirical_coverage"
                        ]
                        and
                        other[
                            "mean_physical_probe_count"
                        ]
                        <=
                        point[
                            "mean_physical_probe_count"
                        ]
                    )

                    strictly_better = (
                        other[
                            "empirical_coverage"
                        ]
                        >
                        point[
                            "empirical_coverage"
                        ]
                        or
                        other[
                            "mean_physical_probe_count"
                        ]
                        <
                        point[
                            "mean_physical_probe_count"
                        ]
                    )

                    if (
                        better_or_equal
                        and
                        strictly_better
                    ):
                        dominated = True
                        break

                if not dominated:
                    pareto.append(
                        point
                    )

            key = (
                f"{mode}|{beam_count}"
            )

            result[
                key
            ] = {
                "reliability_overhead_curve":
                    points,

                "pareto_frontier":
                    pareto,
            }

    return result


def descriptive_slices(
    adaptive_records,
):
    records = [
        row
        for row in adaptive_records
        if (
            row[
                "geometry_mode"
            ]
            ==
            PRIMARY_FORMAL_GEOMETRY
            and
            float(
                row[
                    "requested_probability_mass"
                ]
            )
            ==
            NOMINAL_Q
        )
    ]

    def bin_label(
        value,
        edges,
        suffix,
    ):
        for left, right in zip(
            edges[
                :-1
            ],
            edges[
                1:
            ],
        ):
            if (
                value
                >=
                left
                and
                value
                <
                right
            ):
                right_text = (
                    "inf"
                    if math.isinf(
                        right
                    )
                    else
                    f"{right:g}"
                )

                return (
                    f"[{left:g},{right_text})"
                    f"{suffix}"
                )

        return "OUT_OF_RANGE"

    distance_groups = defaultdict(
        list
    )

    uncertainty_groups = defaultdict(
        list
    )

    for row in records:
        distance_groups[
            bin_label(
                float(
                    row[
                        "receiver_distance_m"
                    ]
                ),
                DISTANCE_BINS_M,
                "m",
            )
        ].append(
            row
        )

        uncertainty_deg = math.degrees(
            float(
                row[
                    "azimuth_uncertainty_std_rad"
                ]
            )
        )

        uncertainty_groups[
            bin_label(
                uncertainty_deg,
                UNCERTAINTY_BINS_DEG,
                "deg",
            )
        ].append(
            row
        )

    def summarize(
        groups,
    ):
        result = {}

        for key, rows in sorted(
            groups.items()
        ):
            result[
                key
            ] = {
                "count":
                    len(
                        rows
                    ),

                "coverage":
                    mean_or_none(
                        1.0
                        if row[
                            "containment_hit"
                        ]
                        else
                        0.0
                        for row in rows
                    ),

                "mean_physical_probe_count":
                    mean_or_none(
                        row[
                            "physical_probe_count"
                        ]
                        for row in rows
                    ),

                "mean_BER":
                    mean_or_none(
                        row[
                            "ber"
                        ]
                        for row in rows
                    ),

                "mean_effective_rate_bps":
                    mean_or_none(
                        row[
                            "effective_rate_bps"
                        ]
                        for row in rows
                    ),
            }

        return result

    return {
        "semantics":
            (
                "fixed_pre_run_descriptive_bins_"
                "not_used_for_acceptance_or_tuning"
            ),

        "distance":
            summarize(
                distance_groups
            ),

        "uncertainty":
            summarize(
                uncertainty_groups
            ),
    }


# ============================================================
# Main
# ============================================================

def main():
    acceptance_policy = (
        verify_prefreeze()
    )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    parameters = (
        frozen_stage5_parameters()
    )

    formal_rows = (
        read_formal_rows()
    )

    validation_rows = (
        read_validation_manifest(
            VALIDATION_MANIFEST
        )
    )

    validation_by_id = {
        str(
            row.scenario_id
        ):
            row
        for row in validation_rows
    }

    runtime = (
        build_gaussian_runtime()
    )

    codebooks = (
        build_codebooks(
            parameters
        )
    )

    script_sha = file_sha256(
        Path(
            __file__
        )
    )

    config_hashes = {
        path.name:
            file_sha256(
                path
            )
        for path in sorted(
            (S5 / "configs")
            .glob("*.json")
        )
    }

    fingerprint_payload = {
        "script_sha256":
            script_sha,

        "formal_manifest_sha256":
            EXPECTED_FORMAL_MANIFEST_SHA,

        "Gaussian_checkpoint_sha256":
            EXPECTED_GAUSSIAN_SHA,

        "normalization_sha256":
            EXPECTED_NORMALIZATION_SHA,

        "calibrator_sha256":
            EXPECTED_CALIBRATOR_SHA,

        "Stage4_handoff_sha256":
            EXPECTED_STAGE4_HANDOFF_SHA,

        "acceptance_policy_sha256":
            EXPECTED_ACCEPTANCE_SHA,

        "stage5_config_sha256":
            config_hashes,

        "primary_geometry_mode":
            PRIMARY_FORMAL_GEOMETRY,

        "formal_N":
            FORMAL_N,
    }

    run_fingerprint = (
        canonical_sha256(
            fingerprint_payload
        )
    )

    clean_config, degraded_config = (
        load_frozen_stage2_configs()
    )

    print(
        "============================================================"
    )
    print(
        "BLOCK 5.8 PART 2/2 — FROZEN FORMAL N=120"
    )
    print(
        "============================================================"
    )
    print(
        "preflight                  = PASS"
    )
    print(
        "formal population          = N=120 / SHA PASS"
    )
    print(
        "Gaussian checkpoint        = SHA PASS"
    )
    print(
        "calibrator                 = SHA PASS"
    )
    print(
        "acceptance q / alpha       = 0.95 / 0.05 FROZEN"
    )
    print(
        "primary formal geometry    =",
        PRIMARY_FORMAL_GEOMETRY,
    )
    print(
        "run fingerprint            =",
        run_fingerprint,
    )

    scene_results = []
    reused = 0

    for rank, formal in enumerate(
        formal_rows,
        start=1,
    ):
        result, was_reused = (
            process_scenario(
                rank=rank,
                formal=formal,
                validation_by_id=(
                    validation_by_id
                ),
                clean_config=clean_config,
                degraded_config=(
                    degraded_config
                ),
                runtime=runtime,
                parameters=parameters,
                codebooks=codebooks,
                run_fingerprint=(
                    run_fingerprint
                ),
            )
        )

        scene_results.append(
            result
        )

        reused += int(
            was_reused
        )

        if (
            rank
            %
            20
            ==
            0
            or
            rank
            ==
            FORMAL_N
        ):
            print(
                f"formal progress             = "
                f"{rank}/{FORMAL_N}"
            )

    if len(
        scene_results
    ) != FORMAL_N:
        raise RuntimeError(
            "Formal scene-result count != 120."
        )

    adaptive_records = []

    baseline_records = []

    availability = {
        "formal_scenarios":
            FORMAL_N,

        "scenarios_with_eligible_communication_receiver":
            0,

        "scenarios_without_eligible_communication_receiver":
            0,

        "selected_receiver_prediction_unavailable":
            0,

        "valid_future_receiver_endpoints":
            {
                str(
                    horizon
                ):
                    0
                for horizon in HORIZONS_S
            },

        "class_resolved_causal_candidates_total":
            0,

        "unresolved_causal_histories_total":
            0,

        "selected_receiver_semantic_class":
            Counter(),
    }

    inference_runtimes = []

    for result in (
        scene_results
    ):
        adaptive_records.extend(
            result[
                "adaptive_records"
            ]
        )

        baseline_records.extend(
            result[
                "baseline_records"
            ]
        )

        availability[
            "class_resolved_causal_candidates_total"
        ] += int(
            result[
                "causal_population"
            ][
                "class_resolved_candidate_count"
            ]
        )

        availability[
            "unresolved_causal_histories_total"
        ] += int(
            result[
                "causal_population"
            ][
                "unresolved_history_count"
            ]
        )

        if bool(
            result[
                "receiver"
            ][
                "selected"
            ]
        ):
            availability[
                "scenarios_with_eligible_communication_receiver"
            ] += 1

            availability[
                "selected_receiver_semantic_class"
            ][
                str(
                    result[
                        "receiver"
                    ][
                        "semantic_class"
                    ]
                )
            ] += 1

        else:
            availability[
                "scenarios_without_eligible_communication_receiver"
            ] += 1

        if (
            result[
                "runtime"
            ][
                "stage4_predictor_inference_runtime_s"
            ]
            is not None
        ):
            inference_runtimes.append(
                float(
                    result[
                        "runtime"
                    ][
                        "stage4_predictor_inference_runtime_s"
                    ]
                )
            )

        for horizon in (
            HORIZONS_S
        ):
            if bool(
                result[
                    "availability"
                ][
                    "valid_future_receiver_endpoint"
                ][
                    str(
                        horizon
                    )
                ]
            ):
                availability[
                    "valid_future_receiver_endpoints"
                ][
                    str(
                        horizon
                    )
                ] += 1

    availability[
        "selected_receiver_semantic_class"
    ] = dict(
        availability[
            "selected_receiver_semantic_class"
        ]
    )

    adaptive_aggregate = (
        aggregate_adaptive(
            adaptive_records
        )
    )

    baseline_aggregate = (
        aggregate_baselines(
            baseline_records
        )
    )

    acceptance = (
        formal_acceptance(
            adaptive_aggregate,
            baseline_aggregate,
        )
    )

    reliability = (
        reliability_overhead_curve(
            adaptive_aggregate
        )
    )

    slices = descriptive_slices(
        adaptive_records
    )

    all_records = []

    for row in adaptive_records:
        all_records.append(
            {
                "record_type":
                    "adaptive",

                **row,
            }
        )

    for row in baseline_records:
        all_records.append(
            {
                "record_type":
                    "baseline",

                **row,
            }
        )

    all_records.sort(
        key=lambda row: (
            row[
                "scenario_id"
            ],
            row[
                "geometry_mode"
            ],
            int(
                row[
                    "codebook_size"
                ]
            ),
            float(
                row[
                    "horizon_s"
                ]
            ),
            row[
                "record_type"
            ],
            str(
                row.get(
                    "policy",
                    "",
                )
            ),
            float(
                row.get(
                    "requested_probability_mass",
                    -1.0,
                )
            ),
        )
    )

    atomic_jsonl(
        MERGED_RECORDS,
        all_records,
    )

    scene_cache_entries = []

    deterministic_digest = sha256()

    for result in scene_results:
        path = (
            CACHE_DIR
            /
            (
                f"{int(result['rank']):03d}_"
                f"{result['scenario_id']}.json"
            )
        )

        deterministic_payload = (
            strip_nondeterministic_runtime(
                result
            )
        )

        deterministic_digest.update(
            json.dumps(
                deterministic_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode(
                "utf-8"
            )
        )

        deterministic_digest.update(
            b"\n"
        )

        scene_cache_entries.append(
            {
                "rank":
                    int(
                        result[
                            "rank"
                        ]
                    ),

                "scenario_id":
                    result[
                        "scenario_id"
                    ],

                "path":
                    str(
                        path
                    ),

                "sha256":
                    file_sha256(
                        path
                    ),
            }
        )

    cache_manifest_payload = {
        "stage":
            5,

        "block":
            "5.8",

        "formal_N":
            FORMAL_N,

        "run_fingerprint":
            run_fingerprint,

        "scene_cache_count":
            len(
                scene_cache_entries
            ),

        "scene_caches":
            scene_cache_entries,

        "deterministic_formal_content_sha256":
            deterministic_digest.hexdigest(),

        "merged_records_path":
            str(
                MERGED_RECORDS
            ),

        "merged_records_sha256":
            file_sha256(
                MERGED_RECORDS
            ),
    }

    atomic_json(
        CACHE_MANIFEST,
        cache_manifest_payload,
    )

    adaptive_primary_nominal = [
        row
        for row in adaptive_records
        if (
            row[
                "geometry_mode"
            ]
            ==
            PRIMARY_FORMAL_GEOMETRY
            and
            float(
                row[
                    "requested_probability_mass"
                ]
            )
            ==
            NOMINAL_Q
        )
    ]

    report = {
        "stage":
            5,

        "block":
            "5.8",

        "phase":
            "frozen_formal_N120_and_PDF_acceptance",

        "status":
            (
                "PASS"
                if acceptance[
                    "stage5_pdf_acceptance_pass"
                ]
                else
                "BLOCKED"
            ),

        "formal_population": {
            "scenario_count":
                FORMAL_N,

            "same_immutable_population_as_Stage3_4":
                True,

            "manifest_path":
                str(
                    FORMAL_MANIFEST
                ),

            "manifest_sha256":
                EXPECTED_FORMAL_MANIFEST_SHA,
        },

        "provenance": {
            "run_fingerprint":
                run_fingerprint,

            "script_sha256":
                script_sha,

            "Gaussian_checkpoint_sha256":
                EXPECTED_GAUSSIAN_SHA,

            "normalization_sha256":
                EXPECTED_NORMALIZATION_SHA,

            "covariance_calibrator_sha256":
                EXPECTED_CALIBRATOR_SHA,

            "Stage4_to_Stage5_handoff_sha256":
                EXPECTED_STAGE4_HANDOFF_SHA,

            "formal_acceptance_policy_sha256":
                EXPECTED_ACCEPTANCE_SHA,

            "stage5_config_sha256":
                config_hashes,

            "receiver_offset_source":
                parameters[
                    "offset"
                ][
                    "source"
                ],

            "codebook_support_source":
                parameters[
                    "support_source"
                ],

            "calibration_variance_scale":
                list(
                    runtime
                    .variance_scale
                ),

            "calibration_variance_scale_source":
                runtime
                .variance_scale_source,

            "formal_cache_manifest_path":
                str(
                    CACHE_MANIFEST
                ),

            "formal_cache_manifest_sha256":
                file_sha256(
                    CACHE_MANIFEST
                ),

            "deterministic_formal_content_sha256":
                cache_manifest_payload[
                    "deterministic_formal_content_sha256"
                ],

            "merged_records_sha256":
                cache_manifest_payload[
                    "merged_records_sha256"
                ],
        },

        "frozen_policy": {
            "primary_receiver_policy":
                "nearest_causal_vehicle_ahead",

            "primary_receiver_geometry_mode":
                PRIMARY_FORMAL_GEOMETRY,

            "receiver_geometry_modes":
                list(
                    GEOMETRY_MODES
                ),

            "codebook_sizes":
                list(
                    CODEBOOK_SIZES
                ),

            "horizons_s":
                list(
                    HORIZONS_S
                ),

            "requested_probability_mass":
                list(
                    REQUESTED_Q
                ),

            "nominal_q":
                NOMINAL_Q,

            "formal_alpha":
                FORMAL_EMPIRICAL_ALPHA,

            "coverage_method":
                acceptance_policy[
                    "formal_empirical_coverage_tolerance"
                ][
                    "method"
                ],

            "MC_sample_count":
                int(
                    parameters[
                        "mc_sample_count"
                    ]
                ),

            "Tbeam_s":
                float(
                    parameters[
                        "Tbeam_s"
                    ]
                ),

            "Tframe_s":
                float(
                    parameters[
                        "Tframe_s"
                    ]
                ),
        },

        "receiver_availability":
            availability,

        "beam_metrics": {
            "adaptive":
                adaptive_aggregate,

            "fixed_and_classical_baselines":
                baseline_aggregate,

            "top_1_hit_rate":
                {
                    key:
                        value[
                            "hit_rate"
                        ]
                    for key, value in (
                        baseline_aggregate.items()
                    )
                    if value[
                        "policy"
                    ]
                    ==
                    "fixed_top_1"
                },

            "top_3_hit_rate":
                {
                    key:
                        value[
                            "hit_rate"
                        ]
                    for key, value in (
                        baseline_aggregate.items()
                    )
                    if value[
                        "policy"
                    ]
                    ==
                    "fixed_top_3"
                },

            "top_k_hit_rate":
                {
                    key:
                        value[
                            "empirical_coverage"
                        ]
                    for key, value in (
                        adaptive_aggregate.items()
                    )
                },

            "reliability_overhead_and_Pareto":
                reliability,

            "distance_and_uncertainty_analysis":
                slices,
        },

        "optical_and_controller_summary": {
            "mean_selected_K_nominal_primary":
                mean_or_none(
                    row[
                        "selected_k"
                    ]
                    for row in (
                        adaptive_primary_nominal
                    )
                ),

            "mean_physical_probe_count_nominal_primary":
                mean_or_none(
                    row[
                        "physical_probe_count"
                    ]
                    for row in (
                        adaptive_primary_nominal
                    )
                ),

            "probing_overhead":
                mean_or_none(
                    row[
                        "probing_overhead_fraction"
                    ]
                    for row in (
                        adaptive_primary_nominal
                    )
                ),

            "overhead_reduction":
                mean_or_none(
                    row[
                        "overhead_reduction_vs_exhaustive"
                    ]
                    for row in (
                        adaptive_primary_nominal
                    )
                ),

            "beam_gain_loss_db":
                mean_or_none(
                    row[
                        "beam_gain_loss_db"
                    ]
                    for row in (
                        adaptive_primary_nominal
                    )
                ),

            "optical_received_power_loss_db":
                mean_or_none(
                    row[
                        "received_power_loss_db"
                    ]
                    for row in (
                        adaptive_primary_nominal
                    )
                ),

            "SNR_loss_db":
                mean_or_none(
                    row[
                        "snr_loss_db"
                    ]
                    for row in (
                        adaptive_primary_nominal
                    )
                ),

            "BER":
                mean_or_none(
                    row[
                        "ber"
                    ]
                    for row in (
                        adaptive_primary_nominal
                    )
                ),

            "effective_rate_bps":
                mean_or_none(
                    row[
                        "effective_rate_bps"
                    ]
                    for row in (
                        adaptive_primary_nominal
                    )
                ),

            "effective_rate_loss_bps":
                mean_or_none(
                    row[
                        "effective_rate_loss_bps"
                    ]
                    for row in (
                        adaptive_primary_nominal
                    )
                ),

            "outage_probability":
                mean_or_none(
                    1.0
                    if row[
                        "beam_outage"
                    ]
                    else
                    0.0
                    for row in (
                        adaptive_primary_nominal
                    )
                ),

            "beam_switching_rate":
                mean_or_none(
                    1.0
                    if row[
                        "primary_switched"
                    ]
                    else
                    0.0
                    for row in (
                        adaptive_primary_nominal
                    )
                ),

            "reacquisition_latency_s":
                mean_or_none(
                    row[
                        "reacquisition_latency_s"
                    ]
                    for row in (
                        adaptive_primary_nominal
                    )
                ),
        },

        "latency": {
            "Stage4_predictor_inference_runtime_s":
                mean_or_none(
                    inference_runtimes
                ),

            "posterior_mapping_runtime_s":
                mean_or_none(
                    row[
                        "posterior_mapping_runtime_s"
                    ]
                    for row in (
                        adaptive_primary_nominal
                    )
                ),

            "beam_probability_runtime_s":
                mean_or_none(
                    row[
                        "beam_probability_runtime_s"
                    ]
                    for row in (
                        adaptive_primary_nominal
                    )
                ),

            "topk_selection_runtime_s":
                mean_or_none(
                    row[
                        "topk_selection_runtime_s"
                    ]
                    for row in (
                        adaptive_primary_nominal
                    )
                ),

            "probing_latency_semantics":
                "physical_probe_count * frozen_Tbeam",

            "reacquisition_latency_reported":
                True,
        },

        "acceptance":
            acceptance,

        "scientific_execution": {
            "Stage4_inference":
                True,

            "Stage5_formal_controller_run":
                True,

            "formal_metrics_computed":
                True,

            "future_truth_controller_input":
                False,

            "tracks_to_predict_receiver_selection":
                False,

            "future_truth_evaluator_only_after_controller_decision":
                True,

            "formal_results_used_for_parameter_selection":
                False,

            "posthoc_tuning":
                False,

            "training":
                False,

            "normalization_refit":
                False,

            "recalibration":
                False,

            "model_selection":
                False,

            "ADB_started":
                False,

            "DeepSense_started":
                False,
        },

        "cache_reuse": {
            "reused_scene_count":
                int(
                    reused
                ),

            "fresh_scene_count":
                int(
                    FORMAL_N
                    -
                    reused
                ),
        },

        "next":
            (
                "Block5.9 exact reproducibility freeze"
                if
                acceptance[
                    "stage5_pdf_acceptance_pass"
                ]
                else
                "Stage5 STATUS=BLOCKED; do not tune "
                "threshold/codebook/posterior on formal N=120"
            ),
    }

    atomic_json(
        REPORT,
        report,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.8 FORMAL RESULT"
    )
    print(
        "============================================================"
    )
    print(
        "formal scenarios           =",
        FORMAL_N,
    )
    print(
        "eligible receivers         =",
        availability[
            "scenarios_with_eligible_communication_receiver"
        ],
    )
    print(
        "no eligible receiver       =",
        availability[
            "scenarios_without_eligible_communication_receiver"
        ],
    )
    print(
        "formal scene caches reused =",
        reused,
    )
    print(
        "coverage acceptance        =",
        "PASS"
        if acceptance[
            "coverage_acceptance_pass"
        ]
        else
        "BLOCKED",
    )
    print(
        "overhead acceptance        =",
        "PASS"
        if acceptance[
            "overhead_reduction_acceptance_pass"
        ]
        else
        "BLOCKED",
    )

    for test in (
        acceptance[
            "overhead_tests"
        ]
    ):
        print(
            "codebook",
            test[
                "codebook_size"
            ],
            "adaptive probes =",
            test[
                "adaptive_mean_physical_probe_count"
            ],
            "best valid fixed =",
            test[
                "best_coverage_valid_fixed_topk"
            ],
            "overhead pass =",
            test[
                "codebook_overhead_pass"
            ],
        )

    print(
        "deterministic content SHA  =",
        cache_manifest_payload[
            "deterministic_formal_content_sha256"
        ],
    )
    print(
        "report                     =",
        REPORT,
    )

    if acceptance[
        "stage5_pdf_acceptance_pass"
    ]:
        print(
            "BLOCK 5.8 STATUS           = PASS"
        )
        print(
            "checker_exit_code          = 0"
        )

        return 0

    print(
        "BLOCK 5.8 STATUS           = BLOCKED"
    )
    print(
        "NO POST-HOC FORMAL TUNING IS ALLOWED."
    )
    print(
        "checker_exit_code          = 3"
    )

    return 3


def guarded_main():
    try:
        return main()

    except Exception as exc:
        failure = {
            "stage":
                5,

            "block":
                "5.8",

            "status":
                "FAIL_IMPLEMENTATION_OR_RUNTIME",

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

            "recovery_hint":
                (
                    "Do not tune any formal parameter. "
                    "Repair only the exact Block5.8 "
                    "implementation/provenance defect and rerun."
                ),
        }

        try:
            atomic_json(
                FAILURE_REPORT,
                failure,
            )

        except Exception:
            pass

        print()
        print(
            "============================================================"
        )
        print(
            "BLOCK 5.8 IMPLEMENTATION/RUNTIME FAILURE"
        )
        print(
            "============================================================"
        )
        print(
            "type    =",
            type(
                exc
            ).__name__,
        )
        print(
            "message =",
            str(
                exc
            ),
        )
        print(
            "failure =",
            FAILURE_REPORT,
        )
        print(
            "checker_exit_code = 2"
        )

        return 2


if __name__ == "__main__":
    raise SystemExit(
        guarded_main()
    )

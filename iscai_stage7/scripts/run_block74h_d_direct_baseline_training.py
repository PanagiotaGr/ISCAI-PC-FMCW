from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

# Must be set before CUDA context creation.
os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from iscai_stage4.models.deterministic_gru import (
    DeterministicGRUConfig,
    DeterministicTrajectoryGRU,
)

from iscai_stage4.ml import (
    CLASS_CYCLIST,
    CLASS_PEDESTRIAN,
    CLASS_VEHICLE,
)

import iscai_stage4.training.normalization as normmod


ROOT = Path("/home/agni/waymo")

S4 = ROOT / "iscai_stage4"
S5 = ROOT / "iscai_stage5"
S6 = ROOT / "iscai_stage6"
S7 = ROOT / "iscai_stage7"

PREREG = (
    S7
    / "configs/"
      "stage7_block74h_c_direct_training_preregistration.json"
)

COMPOSITE = (
    S7
    / "configs/"
      "stage7_block74g_b2_composite_direct_label_authority.json"
)

H_A = (
    S7
    / "reports/block74h/"
      "block74h_a_direct_training_schema_preflight.json"
)

NORMALIZATION = (
    S4
    / "artifacts/block43/"
      "fit_normalization.json"
)

DET_SOURCE = (
    S4
    / "src/iscai_stage4/models/"
      "deterministic_gru.py"
)

NORM_SOURCE = (
    S4
    / "src/iscai_stage4/training/"
      "normalization.py"
)

CACHE_ROOT = (
    S4
    / "artifacts/block43/cache"
)

BEAM_ROOT = (
    S7
    / "artifacts/block74g_b2_r5_beam"
)

ADB_ROOT = (
    S7
    / "artifacts/block74g_b2"
)

BEAM_DIR = (
    S7
    / "artifacts/block74h_direct/beam"
)

ADB_DIR = (
    S7
    / "artifacts/block74h_direct/adb"
)

BEAM_BEST = (
    BEAM_DIR
    / "best.pt"
)

BEAM_RESUME = (
    BEAM_DIR
    / "resume.pt"
)

ADB_BEST = (
    ADB_DIR
    / "best.pt"
)

ADB_RESUME = (
    ADB_DIR
    / "resume.pt"
)

REPORT = (
    S7
    / "reports/block74h/"
      "block74h_d_direct_baseline_training.json"
)


EXPECTED = {
    S4 / "reports/stage4_final_closure.json":
        "570da4feb918b1025b5e85cc919360d922b471c468c85b3844f13fb7774e7c2f",

    S5 / "reports/stage5_final_closure.json":
        "c83731948749f3ca20742aaac6b7474f53c755cbc8209dd31db969d229f60610",

    S6 / "reports/stage6_final_certificate.json":
        "e3870d8c93456f0bf6470467dcbf89ee4e75e7910c6a364c487d2f30e8163cfc",

    S6 / "artifacts/stage6_to_stage7_handoff.json":
        "7456e465043fa85e257fc4322159ee44c008992888aeb8d040d4cfeead9fb449",

    PREREG:
        "fc0726e3c0d964e48b49a63d775070b1487a3716f160813391881d47cccb1333",

    COMPOSITE:
        "acb87c8486fb0cc3a80004c6b08e3c657b6ad680256d77abad226575bd4082fb",

    H_A:
        "52a30de4a83fc7b367717f612b48bbd41d21e76e131fdf6e1699734dada81cba",

    NORMALIZATION:
        "3d7fc0a66d4a4f566f6569befa9c3766ecae21830bb4e46256df2f326a82a5f6",

    DET_SOURCE:
        "7984381e09fe7b3f307b353cffc775941343450d4221969e733a048c9392e9f7",

    NORM_SOURCE:
        "971c45c0fc5d8f8778bf926effee19e5b022772a5f1f5ec518062eadcfe833c0",
}


BEAM_SEED = 2689822245
ADB_SEED = 3910524196

BATCH_SIZE = 512
MAX_EPOCHS = 20
PATIENCE = 4

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1.0e-5
GRADIENT_CLIP = 5.0

CODEBOOKS = (
    16,
    32,
    64,
)

HORIZONS = np.asarray(
    [
        0.1,
        0.3,
        0.5,
        1.0,
    ],
    dtype=np.float32,
)

THETA_MIN = -0.4363323129985824
THETA_STEP = 0.0017453292519943296

RANGE_STEP = 0.5
RANGE_MAX = 150.0

EXPECTED_BEAM_FIT = 2888
EXPECTED_BEAM_DEV = 364
EXPECTED_BEAM_TOTAL = 3252

EXPECTED_ADB_TOTAL = 131500564
EXPECTED_ADB_POSITIVE = 65750282
EXPECTED_ADB_NEGATIVE = 65750282


class FailClosed(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise FailClosed(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def state_dict_sha256(state_dict) -> str:
    h = hashlib.sha256()

    for key in sorted(
        state_dict.keys()
    ):
        tensor = (
            state_dict[key]
            .detach()
            .cpu()
            .contiguous()
        )

        h.update(
            key.encode(
                "utf-8"
            )
        )

        h.update(
            str(
                tensor.dtype
            ).encode(
                "utf-8"
            )
        )

        h.update(
            json.dumps(
                list(
                    tensor.shape
                )
            ).encode(
                "utf-8"
            )
        )

        h.update(
            tensor.numpy().tobytes(
                order="C"
            )
        )

    return h.hexdigest()


def tensor_sha256(
    tensors,
) -> str:
    h = hashlib.sha256()

    for name, tensor in tensors:
        array = (
            tensor.detach()
            .cpu()
            .contiguous()
            .numpy()
        )

        h.update(
            str(name).encode(
                "utf-8"
            )
        )

        h.update(
            str(
                array.dtype
            ).encode(
                "utf-8"
            )
        )

        h.update(
            json.dumps(
                list(
                    array.shape
                )
            ).encode(
                "utf-8"
            )
        )

        h.update(
            array.tobytes(
                order="C"
            )
        )

    return h.hexdigest()


def load_json(path: Path):
    require(
        path.is_file(),
        f"Missing JSON: {path}",
    )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def atomic_json(
    path: Path,
    payload,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
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

    tmp.replace(path)


def atomic_torch_save(
    path: Path,
    payload,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        payload,
        tmp,
    )

    tmp.replace(path)


def stable_u64(
    text: str,
) -> int:
    digest = hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )


def set_global_determinism(
    seed: int,
):
    random.seed(
        int(seed)
    )

    np.random.seed(
        int(seed)
        %
        (2**32)
    )

    torch.manual_seed(
        int(seed)
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            int(seed)
        )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(
        True
    )


def choose_device():
    if torch.cuda.is_available():
        return torch.device(
            "cuda:0"
        )

    return torch.device(
        "cpu"
    )


# ============================================================
# Exact fit-only normalization
# ============================================================

class FitInputNormalizer:
    def __init__(
        self,
        path: Path,
    ):
        payload = load_json(
            path
        )

        require(
            payload.get(
                "source"
            )
            ==
            "fit_only",
            (
                "Normalization source "
                "is not fit-only."
            ),
        )

        require(
            payload.get(
                "development_used"
            )
            is False,
            (
                "Development statistics "
                "entered normalization."
            ),
        )

        require(
            payload.get(
                "calibration_used"
            )
            is False,
            (
                "Calibration statistics "
                "entered normalization."
            ),
        )

        self.history_mean = np.asarray(
            payload[
                "continuous_feature_mean"
            ],
            dtype=np.float32,
        )

        self.history_std = np.asarray(
            payload[
                "continuous_feature_std"
            ],
            dtype=np.float32,
        )

        self.map_mean = np.asarray(
            payload[
                "map_mean"
            ],
            dtype=np.float32,
        )

        self.map_std = np.asarray(
            payload[
                "map_std"
            ],
            dtype=np.float32,
        )

        require(
            self.history_mean.shape
            ==
            (
                normmod
                .CONTINUOUS_HISTORY_DIMS,
            ),
            "History-normalization shape changed.",
        )

        require(
            self.history_std.shape
            ==
            self.history_mean.shape,
            "History std shape changed.",
        )

        require(
            self.map_mean.shape
            ==
            (10,),
            "Map-normalization shape changed.",
        )

        require(
            self.map_std.shape
            ==
            (10,),
            "Map std shape changed.",
        )

        require(
            np.all(
                self.history_std
                >
                0
            ),
            "History normalization std invalid.",
        )

        require(
            np.all(
                self.map_std
                >
                0
            ),
            "Map normalization std invalid.",
        )

    def history(
        self,
        value,
    ):
        x = np.asarray(
            value,
            dtype=np.float32,
        ).copy()

        require(
            x.shape[-1]
            ==
            14,
            "History feature dimension changed.",
        )

        observed = (
            x[
                ...,
                normmod
                .OBSERVED_MASK_INDEX
            ]
            .copy()
        )

        velocity_valid = (
            x[
                ...,
                normmod
                .VELOCITY_MASK_INDEX
            ]
            .copy()
        )

        x[
            ...,
            :
            normmod
            .CONTINUOUS_HISTORY_DIMS
        ] = (
            x[
                ...,
                :
                normmod
                .CONTINUOUS_HISTORY_DIMS
            ]
            -
            self.history_mean
        ) / self.history_std

        for dim in (
            normmod.POSITION_DIMS
            +
            normmod.COVARIANCE_DIMS
        ):
            x[
                ...,
                dim
            ] *= observed

        for dim in (
            normmod.VELOCITY_DIMS
        ):
            x[
                ...,
                dim
            ] *= velocity_valid

        return x.astype(
            np.float32,
            copy=False,
        )

    def map(
        self,
        value,
    ):
        x = np.asarray(
            value,
            dtype=np.float32,
        )

        return (
            (
                x
                -
                self.map_mean
            )
            /
            self.map_std
        ).astype(
            np.float32,
            copy=False,
        )


# ============================================================
# Exact deterministic causal encoder
# ============================================================

class MatchedDeterministicEncoder(
    nn.Module
):
    def __init__(
        self,
    ):
        super().__init__()

        config = (
            DeterministicGRUConfig(
                input_dim=14,
                target_hidden_dim=64,
                neighbor_hidden_dim=32,
                map_hidden_dim=32,
                fused_hidden_dim=128,
                output_horizons=4,
                output_dim=3,
                maximum_neighbors=8,
            )
        )

        source = (
            DeterministicTrajectoryGRU(
                config
            )
        )

        self.config = config

        # Fresh parameters from the exact frozen
        # deterministic encoder architecture.
        self.target_gru = (
            source.target_gru
        )

        self.neighbor_gru = (
            source.neighbor_gru
        )

        self.map_encoder = (
            source.map_encoder
        )

        self.fusion = (
            source.fusion
        )

    def forward(
        self,
        target_history,
        neighbor_histories,
        neighbor_mask,
        map_context,
    ):
        require(
            target_history.ndim == 3,
            "target_history must be [B,T,F].",
        )

        require(
            neighbor_histories.ndim == 4,
            (
                "neighbor_histories must "
                "be [B,N,T,F]."
            ),
        )

        require(
            neighbor_mask.ndim == 2,
            "neighbor_mask must be [B,N].",
        )

        require(
            map_context.ndim == 2,
            "map_context must be [B,M].",
        )

        batch = int(
            target_history.shape[0]
        )

        require(
            target_history.shape[-1]
            ==
            14,
            "Target feature dimension changed.",
        )

        require(
            neighbor_histories.shape[
                0
            ]
            ==
            batch,
            "Batch mismatch.",
        )

        require(
            neighbor_histories.shape[
                1
            ]
            ==
            8,
            "Neighbour count changed.",
        )

        require(
            neighbor_mask.shape
            ==
            (
                batch,
                8,
            ),
            "Neighbour mask shape changed.",
        )

        require(
            map_context.shape
            ==
            (
                batch,
                10,
            ),
            "Map context shape changed.",
        )

        _, target_hidden = (
            self.target_gru(
                target_history
            )
        )

        target_embedding = (
            target_hidden[-1]
        )

        (
            b,
            n,
            t,
            f,
        ) = (
            neighbor_histories.shape
        )

        flat_neighbors = (
            neighbor_histories
            .reshape(
                b * n,
                t,
                f,
            )
        )

        _, neighbor_hidden = (
            self.neighbor_gru(
                flat_neighbors
            )
        )

        neighbor_embedding = (
            neighbor_hidden[-1]
            .reshape(
                b,
                n,
                self.config
                .neighbor_hidden_dim,
            )
        )

        mask = (
            neighbor_mask
            .unsqueeze(-1)
        )

        pooled_neighbor = (
            (
                neighbor_embedding
                *
                mask
            )
            .sum(
                dim=1
            )
            /
            mask.sum(
                dim=1
            )
            .clamp_min(
                1.0
            )
        )

        map_embedding = (
            self.map_encoder(
                map_context
            )
        )

        fused = torch.cat(
            (
                target_embedding,
                pooled_neighbor,
                map_embedding,
            ),
            dim=-1,
        )

        embedding = (
            self.fusion(
                fused
            )
        )

        require(
            embedding.shape
            ==
            (
                batch,
                128,
            ),
            (
                "Frozen deterministic encoder "
                f"output changed: {embedding.shape}"
            ),
        )

        return embedding


class DirectBeamClassifier(
    nn.Module
):
    def __init__(
        self,
    ):
        super().__init__()

        self.encoder = (
            MatchedDeterministicEncoder()
        )

        self.heads = nn.ModuleDict(
            {
                "16":
                    nn.Linear(
                        128,
                        4 * 16,
                    ),

                "32":
                    nn.Linear(
                        128,
                        4 * 32,
                    ),

                "64":
                    nn.Linear(
                        128,
                        4 * 64,
                    ),
            }
        )

    def forward(
        self,
        target,
        neighbors,
        neighbor_mask,
        map_context,
    ):
        embedding = self.encoder(
            target,
            neighbors,
            neighbor_mask,
            map_context,
        )

        batch = int(
            embedding.shape[0]
        )

        return {
            size:
                self.heads[
                    str(size)
                ](
                    embedding
                )
                .reshape(
                    batch,
                    4,
                    size,
                )
            for size in
            CODEBOOKS
        }


class DirectADBField(
    nn.Module
):
    def __init__(
        self,
    ):
        super().__init__()

        self.encoder = (
            MatchedDeterministicEncoder()
        )

        self.decoder = nn.Sequential(
            nn.Linear(
                135,
                64,
            ),
            nn.GELU(),

            nn.Linear(
                64,
                64,
            ),
            nn.GELU(),

            nn.Linear(
                64,
                1,
            ),
        )

    def encode(
        self,
        target,
        neighbors,
        neighbor_mask,
        map_context,
    ):
        return self.encoder(
            target,
            neighbors,
            neighbor_mask,
            map_context,
        )

    def decode(
        self,
        embedding,
        theta_index,
        range_index,
        horizon_index,
        class_id,
    ):
        theta = (
            THETA_MIN
            +
            theta_index.to(
                dtype=embedding.dtype
            )
            *
            THETA_STEP
        )

        sin_theta = torch.sin(
            theta
        ).unsqueeze(
            -1
        )

        cos_theta = torch.cos(
            theta
        ).unsqueeze(
            -1
        )

        normalized_range = (
            range_index.to(
                dtype=embedding.dtype
            )
            *
            RANGE_STEP
            /
            RANGE_MAX
        ).unsqueeze(
            -1
        )

        horizon_values = torch.tensor(
            HORIZONS,
            dtype=embedding.dtype,
            device=embedding.device,
        )

        normalized_horizon = (
            horizon_values[
                horizon_index
            ]
            /
            1.0
        ).unsqueeze(
            -1
        )

        class_index = torch.full_like(
            class_id,
            fill_value=-1,
            dtype=torch.long,
        )

        class_index[
            class_id
            ==
            int(
                CLASS_VEHICLE
            )
        ] = 0

        class_index[
            class_id
            ==
            int(
                CLASS_PEDESTRIAN
            )
        ] = 1

        class_index[
            class_id
            ==
            int(
                CLASS_CYCLIST
            )
        ] = 2

        require(
            bool(
                torch.all(
                    class_index
                    >=
                    0
                ).item()
            ),
            "Non-V/P/C class entered ADB decoder.",
        )

        one_hot = F.one_hot(
            class_index,
            num_classes=3,
        ).to(
            dtype=embedding.dtype
        )

        decoder_input = torch.cat(
            (
                embedding,
                sin_theta,
                cos_theta,
                normalized_range,
                normalized_horizon,
                one_hot,
            ),
            dim=-1,
        )

        require(
            decoder_input.shape[-1]
            ==
            135,
            "ADB decoder input dimension changed.",
        )

        return (
            self.decoder(
                decoder_input
            )
            .squeeze(
                -1
            )
        )


# ============================================================
# Beam corpus
# ============================================================

@dataclass(frozen=True)
class BeamArrays:
    target: np.ndarray
    neighbors: np.ndarray
    neighbor_mask: np.ndarray
    map_context: np.ndarray
    valid: np.ndarray
    labels16: np.ndarray
    labels32: np.ndarray
    labels64: np.ndarray


def build_beam_arrays(
    partition: str,
    normalizer,
):
    overlay_dir = (
        BEAM_ROOT
        /
        partition
    )

    files = sorted(
        overlay_dir.glob(
            "*.npz"
        )
    )

    targets = []
    neighbors = []
    neighbor_masks = []
    maps = []

    valids = []
    labels16 = []
    labels32 = []
    labels64 = []

    for path in files:
        with np.load(
            path,
            allow_pickle=False,
        ) as beam:
            available = bool(
                beam[
                    "selected_prediction_available"
                ].item()
            )

            if not available:
                continue

            row = int(
                beam[
                    "beam_row_index"
                ].item()
            )

            source = Path(
                str(
                    beam[
                        "source_cache_path"
                    ].item()
                )
            )

            require(
                source.is_file(),
                f"Missing Stage4 cache: {source}",
            )

            require(
                sha256_file(
                    source
                )
                ==
                str(
                    beam[
                        "source_cache_sha256"
                    ].item()
                ),
                "Beam source-cache SHA mismatch.",
            )

            with np.load(
                source,
                allow_pickle=False,
            ) as cache:
                n = int(
                    cache[
                        "target"
                    ].shape[0]
                )

                require(
                    0 <= row < n,
                    "Beam row index outside cache.",
                )

                targets.append(
                    normalizer.history(
                        cache[
                            "target"
                        ][
                            row
                        ]
                    )
                )

                neighbors.append(
                    normalizer.history(
                        cache[
                            "neighbors"
                        ][
                            row
                        ]
                    )
                )

                neighbor_masks.append(
                    np.asarray(
                        cache[
                            "neighbor_mask"
                        ][
                            row
                        ],
                        dtype=np.float32,
                    )
                )

                maps.append(
                    normalizer.map(
                        cache[
                            "map_context"
                        ][
                            row
                        ]
                    )
                )

            valids.append(
                np.asarray(
                    beam[
                        "beam_valid_mask"
                    ],
                    dtype=bool,
                )
            )

            labels16.append(
                np.asarray(
                    beam[
                        "beam_label_16"
                    ],
                    dtype=np.int64,
                )
            )

            labels32.append(
                np.asarray(
                    beam[
                        "beam_label_32"
                    ],
                    dtype=np.int64,
                )
            )

            labels64.append(
                np.asarray(
                    beam[
                        "beam_label_64"
                    ],
                    dtype=np.int64,
                )
            )

    result = BeamArrays(
        target=np.stack(
            targets,
            axis=0,
        ).astype(
            np.float32
        ),

        neighbors=np.stack(
            neighbors,
            axis=0,
        ).astype(
            np.float32
        ),

        neighbor_mask=np.stack(
            neighbor_masks,
            axis=0,
        ).astype(
            np.float32
        ),

        map_context=np.stack(
            maps,
            axis=0,
        ).astype(
            np.float32
        ),

        valid=np.stack(
            valids,
            axis=0,
        ).astype(
            bool
        ),

        labels16=np.stack(
            labels16,
            axis=0,
        ).astype(
            np.int64
        ),

        labels32=np.stack(
            labels32,
            axis=0,
        ).astype(
            np.int64
        ),

        labels64=np.stack(
            labels64,
            axis=0,
        ).astype(
            np.int64
        ),
    )

    expected = (
        EXPECTED_BEAM_FIT
        if partition == "fit"
        else EXPECTED_BEAM_DEV
    )

    require(
        result.target.shape[0]
        ==
        expected,
        (
            f"{partition}: direct-beam "
            f"examples={result.target.shape[0]}"
        ),
    )

    return result


def beam_batch_to_device(
    arrays: BeamArrays,
    indices,
    device,
):
    idx = np.asarray(
        indices,
        dtype=np.int64,
    )

    return {
        "target":
            torch.from_numpy(
                arrays.target[idx]
            ).to(
                device=device,
            ),

        "neighbors":
            torch.from_numpy(
                arrays.neighbors[idx]
            ).to(
                device=device,
            ),

        "neighbor_mask":
            torch.from_numpy(
                arrays.neighbor_mask[idx]
            ).to(
                device=device,
            ),

        "map_context":
            torch.from_numpy(
                arrays.map_context[idx]
            ).to(
                device=device,
            ),

        "valid":
            torch.from_numpy(
                arrays.valid[idx]
            ).to(
                device=device,
            ),

        "labels": {
            16:
                torch.from_numpy(
                    arrays.labels16[idx]
                ).to(
                    device=device,
                ),

            32:
                torch.from_numpy(
                    arrays.labels32[idx]
                ).to(
                    device=device,
                ),

            64:
                torch.from_numpy(
                    arrays.labels64[idx]
                ).to(
                    device=device,
                ),
        },
    }


def beam_loss_sum_count(
    outputs,
    batch,
):
    loss_sum = torch.zeros(
        (),
        dtype=torch.float32,
        device=(
            batch[
                "target"
            ].device
        ),
    )

    count = 0

    valid = batch[
        "valid"
    ].bool()

    for size in CODEBOOKS:
        logits = outputs[
            size
        ]

        labels = batch[
            "labels"
        ][
            size
        ]

        if bool(
            torch.any(
                valid
            ).item()
        ):
            selected_logits = (
                logits[
                    valid
                ]
            )

            selected_labels = (
                labels[
                    valid
                ]
            )

            loss_sum = (
                loss_sum
                +
                F.cross_entropy(
                    selected_logits,
                    selected_labels,
                    reduction="sum",
                )
            )

            count += int(
                selected_labels.numel()
            )

    require(
        count > 0,
        "Beam batch has zero valid labels.",
    )

    return (
        loss_sum,
        count,
    )


@torch.no_grad()
def evaluate_beam(
    model,
    arrays,
    device,
):
    model.eval()

    total = 0.0
    count = 0

    n = int(
        arrays.target.shape[0]
    )

    for start in range(
        0,
        n,
        BATCH_SIZE,
    ):
        stop = min(
            start + BATCH_SIZE,
            n,
        )

        batch = beam_batch_to_device(
            arrays,
            np.arange(
                start,
                stop,
                dtype=np.int64,
            ),
            device,
        )

        outputs = model(
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

        loss_sum, batch_count = (
            beam_loss_sum_count(
                outputs,
                batch,
            )
        )

        total += float(
            loss_sum.item()
        )

        count += int(
            batch_count
        )

    require(
        count > 0,
        "Beam development set has zero labels.",
    )

    return (
        total
        /
        count
    )


# ============================================================
# ADB corpus
# ============================================================

@dataclass(frozen=True)
class ADBShard:
    label_path: Path
    cache_path: Path
    scenario_id: str
    observation_count: int
    positive_count: int
    negative_count: int


def scan_adb_partition(
    partition: str,
):
    files = sorted(
        (
            ADB_ROOT
            /
            partition
        ).glob(
            "*.npz"
        )
    )

    expected_scenes = (
        4096
        if partition == "fit"
        else 512
    )

    require(
        len(files)
        ==
        expected_scenes,
        (
            f"{partition}: ADB shard "
            f"count={len(files)}"
        ),
    )

    result = []

    for rank, path in enumerate(
        files,
        start=1,
    ):
        with np.load(
            path,
            allow_pickle=False,
        ) as d:
            row = np.asarray(
                d[
                    "adb_obs_row_index"
                ],
                dtype=np.int64,
            )

            horizon = np.asarray(
                d[
                    "adb_obs_horizon_index"
                ],
                dtype=np.int64,
            )

            theta = np.asarray(
                d[
                    "adb_obs_theta_index"
                ],
                dtype=np.int64,
            )

            radial = np.asarray(
                d[
                    "adb_obs_range_index"
                ],
                dtype=np.int64,
            )

            target = np.asarray(
                d[
                    "adb_obs_target"
                ],
                dtype=np.uint8,
            )

            count = int(
                target.size
            )

            require(
                row.shape
                ==
                horizon.shape
                ==
                theta.shape
                ==
                radial.shape
                ==
                target.shape,
                "ADB observation arrays differ in shape.",
            )

            require(
                np.all(
                    (
                        target == 0
                    )
                    |
                    (
                        target == 1
                    )
                ),
                "ADB target not binary.",
            )

            require(
                np.all(
                    (
                        horizon >= 0
                    )
                    &
                    (
                        horizon < 4
                    )
                ),
                "ADB horizon index invalid.",
            )

            require(
                np.all(
                    (
                        theta >= 0
                    )
                    &
                    (
                        theta < 501
                    )
                ),
                "ADB theta index invalid.",
            )

            require(
                np.all(
                    (
                        radial >= 0
                    )
                    &
                    (
                        radial < 301
                    )
                ),
                "ADB range index invalid.",
            )

            sample_count = int(
                d[
                    "sample_count"
                ].item()
            )

            if count > 0:
                require(
                    int(
                        row.min()
                    )
                    >=
                    0
                    and
                    int(
                        row.max()
                    )
                    <
                    sample_count,
                    "ADB row index invalid.",
                )

            source = Path(
                str(
                    d[
                        "source_cache_path"
                    ].item()
                )
            )

            require(
                source.is_file(),
                (
                    "ADB Stage4 source cache "
                    f"missing: {source}"
                ),
            )

            require(
                sha256_file(
                    source
                )
                ==
                str(
                    d[
                        "source_cache_sha256"
                    ].item()
                ),
                "ADB source-cache SHA mismatch.",
            )

            positive = int(
                np.count_nonzero(
                    target == 1
                )
            )

            negative = (
                count
                -
                positive
            )

            result.append(
                ADBShard(
                    label_path=path,
                    cache_path=source,
                    scenario_id=str(
                        d[
                            "scenario_id"
                        ].item()
                    ),
                    observation_count=count,
                    positive_count=positive,
                    negative_count=negative,
                )
            )

        if (
            rank % 500 == 0
            or
            rank == expected_scenes
        ):
            print(
                (
                    f"ADB scan {partition}: "
                    f"{rank}/{expected_scenes}"
                ),
                flush=True,
            )

    return result


def load_adb_cache(
    shard: ADBShard,
    normalizer,
):
    with np.load(
        shard.cache_path,
        allow_pickle=False,
    ) as cache:
        target = normalizer.history(
            cache[
                "target"
            ]
        )

        neighbors = normalizer.history(
            cache[
                "neighbors"
            ]
        )

        neighbor_mask = np.asarray(
            cache[
                "neighbor_mask"
            ],
            dtype=np.float32,
        )

        map_context = normalizer.map(
            cache[
                "map_context"
            ]
        )

        class_id = np.asarray(
            cache[
                "class_id"
            ],
            dtype=np.int64,
        )

    return (
        target,
        neighbors,
        neighbor_mask,
        map_context,
        class_id,
    )


def load_adb_observations(
    shard: ADBShard,
):
    with np.load(
        shard.label_path,
        allow_pickle=False,
    ) as d:
        return {
            "row":
                np.asarray(
                    d[
                        "adb_obs_row_index"
                    ],
                    dtype=np.int64,
                ),

            "horizon":
                np.asarray(
                    d[
                        "adb_obs_horizon_index"
                    ],
                    dtype=np.int64,
                ),

            "theta":
                np.asarray(
                    d[
                        "adb_obs_theta_index"
                    ],
                    dtype=np.int64,
                ),

            "range":
                np.asarray(
                    d[
                        "adb_obs_range_index"
                    ],
                    dtype=np.int64,
                ),

            "target":
                np.asarray(
                    d[
                        "adb_obs_target"
                    ],
                    dtype=np.float32,
                ),
        }


def adb_train_batch(
    *,
    model,
    optimizer,
    cache_arrays,
    observations,
    observation_indices,
    device,
):
    (
        target_all,
        neighbors_all,
        neighbor_mask_all,
        map_all,
        class_all,
    ) = cache_arrays

    obs_idx = np.asarray(
        observation_indices,
        dtype=np.int64,
    )

    rows = observations[
        "row"
    ][
        obs_idx
    ]

    unique_rows, inverse = np.unique(
        rows,
        return_inverse=True,
    )

    target = torch.from_numpy(
        target_all[
            unique_rows
        ]
    ).to(
        device=device
    )

    neighbors = torch.from_numpy(
        neighbors_all[
            unique_rows
        ]
    ).to(
        device=device
    )

    neighbor_mask = torch.from_numpy(
        neighbor_mask_all[
            unique_rows
        ]
    ).to(
        device=device
    )

    map_context = torch.from_numpy(
        map_all[
            unique_rows
        ]
    ).to(
        device=device
    )

    embedding_unique = model.encode(
        target,
        neighbors,
        neighbor_mask,
        map_context,
    )

    inverse_t = torch.from_numpy(
        inverse.astype(
            np.int64
        )
    ).to(
        device=device
    )

    embedding = embedding_unique[
        inverse_t
    ]

    horizon = torch.from_numpy(
        observations[
            "horizon"
        ][
            obs_idx
        ]
    ).to(
        device=device,
        dtype=torch.long,
    )

    theta = torch.from_numpy(
        observations[
            "theta"
        ][
            obs_idx
        ]
    ).to(
        device=device,
        dtype=torch.long,
    )

    radial = torch.from_numpy(
        observations[
            "range"
        ][
            obs_idx
        ]
    ).to(
        device=device,
        dtype=torch.long,
    )

    class_id = torch.from_numpy(
        class_all[
            rows
        ]
    ).to(
        device=device,
        dtype=torch.long,
    )

    target_value = torch.from_numpy(
        observations[
            "target"
        ][
            obs_idx
        ]
    ).to(
        device=device,
        dtype=torch.float32,
    )

    logits = model.decode(
        embedding,
        theta,
        radial,
        horizon,
        class_id,
    )

    require(
        logits.shape
        ==
        target_value.shape,
        "ADB logits/target shape mismatch.",
    )

    loss = (
        F.binary_cross_entropy_with_logits(
            logits,
            target_value,
            reduction="mean",
        )
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=GRADIENT_CLIP,
    )

    optimizer.step()

    return (
        float(
            loss.item()
        )
        *
        int(
            target_value.numel()
        ),
        int(
            target_value.numel()
        ),
    )


@torch.no_grad()
def evaluate_adb(
    model,
    shards,
    normalizer,
    device,
):
    model.eval()

    total_loss = 0.0
    total_count = 0

    for shard_index, shard in enumerate(
        shards,
        start=1,
    ):
        if (
            shard.observation_count
            ==
            0
        ):
            continue

        cache_arrays = load_adb_cache(
            shard,
            normalizer,
        )

        observations = (
            load_adb_observations(
                shard
            )
        )

        (
            target_all,
            neighbors_all,
            neighbor_mask_all,
            map_all,
            class_all,
        ) = cache_arrays

        target = torch.from_numpy(
            target_all
        ).to(
            device=device
        )

        neighbors = torch.from_numpy(
            neighbors_all
        ).to(
            device=device
        )

        neighbor_mask = torch.from_numpy(
            neighbor_mask_all
        ).to(
            device=device
        )

        map_context = torch.from_numpy(
            map_all
        ).to(
            device=device
        )

        embedding_all = model.encode(
            target,
            neighbors,
            neighbor_mask,
            map_context,
        )

        count = int(
            shard.observation_count
        )

        for start in range(
            0,
            count,
            BATCH_SIZE,
        ):
            stop = min(
                start + BATCH_SIZE,
                count,
            )

            idx = np.arange(
                start,
                stop,
                dtype=np.int64,
            )

            rows = torch.from_numpy(
                observations[
                    "row"
                ][
                    idx
                ]
            ).to(
                device=device,
                dtype=torch.long,
            )

            horizon = torch.from_numpy(
                observations[
                    "horizon"
                ][
                    idx
                ]
            ).to(
                device=device,
                dtype=torch.long,
            )

            theta = torch.from_numpy(
                observations[
                    "theta"
                ][
                    idx
                ]
            ).to(
                device=device,
                dtype=torch.long,
            )

            radial = torch.from_numpy(
                observations[
                    "range"
                ][
                    idx
                ]
            ).to(
                device=device,
                dtype=torch.long,
            )

            class_id = torch.from_numpy(
                class_all[
                    observations[
                        "row"
                    ][
                        idx
                    ]
                ]
            ).to(
                device=device,
                dtype=torch.long,
            )

            target_value = torch.from_numpy(
                observations[
                    "target"
                ][
                    idx
                ]
            ).to(
                device=device,
                dtype=torch.float32,
            )

            logits = model.decode(
                embedding_all[
                    rows
                ],
                theta,
                radial,
                horizon,
                class_id,
            )

            loss_sum = (
                F.binary_cross_entropy_with_logits(
                    logits,
                    target_value,
                    reduction="sum",
                )
            )

            total_loss += float(
                loss_sum.item()
            )

            total_count += int(
                target_value.numel()
            )

        if (
            shard_index % 100 == 0
            or
            shard_index == len(shards)
        ):
            print(
                (
                    "ADB development: "
                    f"{shard_index}/"
                    f"{len(shards)}"
                ),
                flush=True,
            )

    require(
        total_count > 0,
        "ADB development observations are zero.",
    )

    return (
        total_loss
        /
        total_count
    )


# ============================================================
# Checkpoint helpers
# ============================================================

def save_best(
    *,
    path,
    task,
    model,
    epoch,
    dev_loss,
):
    state = {
        key:
            value.detach().cpu()
        for key, value
        in model.state_dict().items()
    }

    payload = {
        "stage":
            7,

        "block":
            "7.4H-D",

        "task":
            task,

        "preregistration_sha256":
            sha256_file(
                PREREG
            ),

        "epoch":
            int(epoch),

        "development_task_loss":
            float(dev_loss),

        "state_dict":
            state,

        "state_dict_sha256":
            state_dict_sha256(
                state
            ),

        "formal_N120_raw_data_opened":
            False,

        "calibration_partition_opened":
            False,

        "post_outcome_tuning":
            False,
    }

    atomic_torch_save(
        path,
        payload,
    )


def load_model_checkpoint(
    model,
    path,
    device,
):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    require(
        checkpoint[
            "preregistration_sha256"
        ]
        ==
        sha256_file(
            PREREG
        ),
        "Checkpoint preregistration mismatch.",
    )

    require(
        state_dict_sha256(
            checkpoint[
                "state_dict"
            ]
        )
        ==
        checkpoint[
            "state_dict_sha256"
        ],
        "Checkpoint state_dict hash mismatch.",
    )

    model.load_state_dict(
        checkpoint[
            "state_dict"
        ],
        strict=True,
    )

    return checkpoint


def parameter_count(
    model,
):
    return int(
        sum(
            parameter.numel()
            for parameter
            in model.parameters()
            if parameter.requires_grad
        )
    )


# ============================================================
# Beam training
# ============================================================

def train_beam(
    *,
    fit,
    dev,
    device,
):
    print()
    print("=" * 78)
    print("DIRECT BEAM TRAINING")
    print("=" * 78)

    set_global_determinism(
        BEAM_SEED
    )

    model = (
        DirectBeamClassifier()
        .to(
            device
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    history = []

    start_epoch = 1
    best_loss = None
    best_epoch = None
    patience = 0
    optimizer_steps = 0

    if BEAM_RESUME.is_file():
        resume = torch.load(
            BEAM_RESUME,
            map_location=device,
            weights_only=False,
        )

        require(
            resume.get(
                "task"
            )
            ==
            "direct_beam",
            "Beam resume task mismatch.",
        )

        require(
            resume.get(
                "preregistration_sha256"
            )
            ==
            sha256_file(
                PREREG
            ),
            "Beam resume preregistration mismatch.",
        )

        model.load_state_dict(
            resume[
                "model_state"
            ],
            strict=True,
        )

        optimizer.load_state_dict(
            resume[
                "optimizer_state"
            ]
        )

        history = list(
            resume[
                "history"
            ]
        )

        start_epoch = int(
            resume[
                "next_epoch"
            ]
        )

        best_loss = float(
            resume[
                "best_loss"
            ]
        )

        best_epoch = int(
            resume[
                "best_epoch"
            ]
        )

        patience = int(
            resume[
                "patience"
            ]
        )

        optimizer_steps = int(
            resume[
                "optimizer_steps"
            ]
        )

        print(
            (
                "beam resume = YES | "
                f"next_epoch={start_epoch}"
            )
        )

    else:
        initial = evaluate_beam(
            model,
            dev,
            device,
        )

        require(
            math.isfinite(
                initial
            ),
            "Initial beam dev loss non-finite.",
        )

        best_loss = float(
            initial
        )

        best_epoch = 0

        history.append(
            {
                "epoch":
                    0,

                "train_loss":
                    None,

                "development_loss":
                    float(
                        initial
                    ),
            }
        )

        save_best(
            path=BEAM_BEST,
            task="direct_beam",
            model=model,
            epoch=0,
            dev_loss=initial,
        )

        print(
            (
                "beam epoch 00 | "
                f"devCE={initial:.9f}"
            )
        )

    n = int(
        fit.target.shape[0]
    )

    for epoch in range(
        start_epoch,
        MAX_EPOCHS + 1,
    ):
        model.train()

        rng = np.random.default_rng(
            BEAM_SEED
            +
            epoch
        )

        order = rng.permutation(
            n
        )

        epoch_loss_sum = 0.0
        epoch_count = 0

        for start in range(
            0,
            n,
            BATCH_SIZE,
        ):
            stop = min(
                start + BATCH_SIZE,
                n,
            )

            batch = beam_batch_to_device(
                fit,
                order[
                    start:
                    stop
                ],
                device,
            )

            outputs = model(
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

            loss_sum, count = (
                beam_loss_sum_count(
                    outputs,
                    batch,
                )
            )

            loss = (
                loss_sum
                /
                count
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRADIENT_CLIP,
            )

            optimizer.step()

            optimizer_steps += 1

            epoch_loss_sum += float(
                loss_sum.item()
            )

            epoch_count += int(
                count
            )

        train_loss = (
            epoch_loss_sum
            /
            epoch_count
        )

        dev_loss = evaluate_beam(
            model,
            dev,
            device,
        )

        require(
            math.isfinite(
                dev_loss
            ),
            "Beam development CE non-finite.",
        )

        improved = (
            dev_loss
            <
            best_loss
        )

        if improved:
            best_loss = float(
                dev_loss
            )

            best_epoch = int(
                epoch
            )

            patience = 0

            save_best(
                path=BEAM_BEST,
                task="direct_beam",
                model=model,
                epoch=epoch,
                dev_loss=dev_loss,
            )

        else:
            patience += 1

        history.append(
            {
                "epoch":
                    int(epoch),

                "train_loss":
                    float(
                        train_loss
                    ),

                "development_loss":
                    float(
                        dev_loss
                    ),
            }
        )

        print(
            (
                f"beam epoch {epoch:02d} | "
                f"trainCE={train_loss:.9f} | "
                f"devCE={dev_loss:.9f} | "
                f"best={best_loss:.9f} | "
                f"bestEpoch={best_epoch} | "
                f"patience={patience}/{PATIENCE}"
            ),
            flush=True,
        )

        atomic_torch_save(
            BEAM_RESUME,
            {
                "task":
                    "direct_beam",

                "preregistration_sha256":
                    sha256_file(
                        PREREG
                    ),

                "next_epoch":
                    int(
                        epoch + 1
                    ),

                "best_loss":
                    float(
                        best_loss
                    ),

                "best_epoch":
                    int(
                        best_epoch
                    ),

                "patience":
                    int(
                        patience
                    ),

                "optimizer_steps":
                    int(
                        optimizer_steps
                    ),

                "history":
                    history,

                "model_state":
                    {
                        key:
                            value.detach().cpu()
                        for key, value
                        in model.state_dict().items()
                    },

                "optimizer_state":
                    optimizer.state_dict(),
            },
        )

        if patience >= PATIENCE:
            print(
                "beam early stopping = TRIGGERED"
            )
            break

    require(
        optimizer_steps > 0,
        "Beam optimizer never stepped.",
    )

    best = load_model_checkpoint(
        model,
        BEAM_BEST,
        device,
    )

    final_best_loss = evaluate_beam(
        model,
        dev,
        device,
    )

    require(
        abs(
            final_best_loss
            -
            float(
                best[
                    "development_task_loss"
                ]
            )
        )
        <=
        1.0e-10,
        "Beam best-checkpoint dev loss mismatch.",
    )

    return {
        "seed":
            BEAM_SEED,

        "train_examples":
            int(
                fit.target.shape[0]
            ),

        "development_examples":
            int(
                dev.target.shape[0]
            ),

        "initial_development_loss":
            float(
                history[0][
                    "development_loss"
                ]
            ),

        "best_epoch":
            int(
                best[
                    "epoch"
                ]
            ),

        "best_development_loss":
            float(
                best[
                    "development_task_loss"
                ]
            ),

        "optimizer_steps":
            int(
                optimizer_steps
            ),

        "history":
            history,

        "trainable_parameter_count":
            parameter_count(
                model
            ),

        "checkpoint_path":
            str(
                BEAM_BEST
            ),

        "checkpoint_sha256":
            sha256_file(
                BEAM_BEST
            ),

        "state_dict_sha256":
            best[
                "state_dict_sha256"
            ],
    }


# ============================================================
# ADB training
# ============================================================

def save_adb_resume(
    *,
    model,
    optimizer,
    epoch,
    shard_position,
    batch_position,
    epoch_loss_sum,
    epoch_count,
    global_steps,
    best_loss,
    best_epoch,
    patience,
    history,
):
    atomic_torch_save(
        ADB_RESUME,
        {
            "task":
                "direct_ADB",

            "preregistration_sha256":
                sha256_file(
                    PREREG
                ),

            "epoch":
                int(epoch),

            "shard_position":
                int(
                    shard_position
                ),

            "batch_position":
                int(
                    batch_position
                ),

            "epoch_loss_sum":
                float(
                    epoch_loss_sum
                ),

            "epoch_count":
                int(
                    epoch_count
                ),

            "global_steps":
                int(
                    global_steps
                ),

            "best_loss":
                float(
                    best_loss
                ),

            "best_epoch":
                int(
                    best_epoch
                ),

            "patience":
                int(
                    patience
                ),

            "history":
                history,

            "model_state":
                {
                    key:
                        value.detach().cpu()
                    for key, value
                    in model.state_dict().items()
                },

            "optimizer_state":
                optimizer.state_dict(),

            "torch_rng_state":
                torch.get_rng_state(),

            "cuda_rng_state_all":
                (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else
                    None
                ),
        },
    )


def train_adb(
    *,
    fit_shards,
    dev_shards,
    normalizer,
    device,
):
    print()
    print("=" * 78)
    print("DIRECT ADB TRAINING")
    print("=" * 78)

    set_global_determinism(
        ADB_SEED
    )

    model = (
        DirectADBField()
        .to(
            device
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    history = []

    epoch = 1
    shard_position = 0
    batch_position = 0

    epoch_loss_sum = 0.0
    epoch_count = 0

    global_steps = 0

    best_loss = None
    best_epoch = None
    patience = 0

    if ADB_RESUME.is_file():
        resume = torch.load(
            ADB_RESUME,
            map_location=device,
            weights_only=False,
        )

        require(
            resume.get(
                "task"
            )
            ==
            "direct_ADB",
            "ADB resume task mismatch.",
        )

        require(
            resume.get(
                "preregistration_sha256"
            )
            ==
            sha256_file(
                PREREG
            ),
            "ADB resume preregistration mismatch.",
        )

        model.load_state_dict(
            resume[
                "model_state"
            ],
            strict=True,
        )

        optimizer.load_state_dict(
            resume[
                "optimizer_state"
            ]
        )

        epoch = int(
            resume[
                "epoch"
            ]
        )

        shard_position = int(
            resume[
                "shard_position"
            ]
        )

        batch_position = int(
            resume[
                "batch_position"
            ]
        )

        epoch_loss_sum = float(
            resume[
                "epoch_loss_sum"
            ]
        )

        epoch_count = int(
            resume[
                "epoch_count"
            ]
        )

        global_steps = int(
            resume[
                "global_steps"
            ]
        )

        best_loss = float(
            resume[
                "best_loss"
            ]
        )

        best_epoch = int(
            resume[
                "best_epoch"
            ]
        )

        patience = int(
            resume[
                "patience"
            ]
        )

        history = list(
            resume[
                "history"
            ]
        )

        torch.set_rng_state(
            resume[
                "torch_rng_state"
            ]
        )

        if (
            torch.cuda.is_available()
            and
            resume[
                "cuda_rng_state_all"
            ]
            is not None
        ):
            torch.cuda.set_rng_state_all(
                resume[
                    "cuda_rng_state_all"
                ]
            )

        print(
            (
                "ADB resume = YES | "
                f"epoch={epoch} | "
                f"shard_position={shard_position} | "
                f"batch_position={batch_position}"
            )
        )

    else:
        initial = evaluate_adb(
            model,
            dev_shards,
            normalizer,
            device,
        )

        require(
            math.isfinite(
                initial
            ),
            "Initial ADB dev BCE non-finite.",
        )

        best_loss = float(
            initial
        )

        best_epoch = 0

        history.append(
            {
                "epoch":
                    0,

                "train_loss":
                    None,

                "development_loss":
                    float(
                        initial
                    ),
            }
        )

        save_best(
            path=ADB_BEST,
            task="direct_ADB",
            model=model,
            epoch=0,
            dev_loss=initial,
        )

        print(
            (
                "ADB epoch 00 | "
                f"devBCE={initial:.9f}"
            )
        )

        save_adb_resume(
            model=model,
            optimizer=optimizer,
            epoch=1,
            shard_position=0,
            batch_position=0,
            epoch_loss_sum=0.0,
            epoch_count=0,
            global_steps=0,
            best_loss=best_loss,
            best_epoch=best_epoch,
            patience=0,
            history=history,
        )

    while epoch <= MAX_EPOCHS:
        model.train()

        shard_rng = np.random.default_rng(
            ADB_SEED
            +
            epoch
        )

        shard_order = (
            shard_rng.permutation(
                len(
                    fit_shards
                )
            )
        )

        for position in range(
            shard_position,
            len(
                shard_order
            ),
        ):
            shard = fit_shards[
                int(
                    shard_order[
                        position
                    ]
                )
            ]

            if (
                shard.observation_count
                ==
                0
            ):
                batch_position = 0

                save_adb_resume(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    shard_position=position + 1,
                    batch_position=0,
                    epoch_loss_sum=epoch_loss_sum,
                    epoch_count=epoch_count,
                    global_steps=global_steps,
                    best_loss=best_loss,
                    best_epoch=best_epoch,
                    patience=patience,
                    history=history,
                )

                continue

            cache_arrays = load_adb_cache(
                shard,
                normalizer,
            )

            observations = (
                load_adb_observations(
                    shard
                )
            )

            permutation_seed = stable_u64(
                (
                    f"{ADB_SEED}|"
                    f"{epoch}|"
                    f"{shard.scenario_id}|"
                    "ADB_OBS"
                )
            )

            observation_rng = (
                np.random.default_rng(
                    permutation_seed
                )
            )

            observation_order = (
                observation_rng.permutation(
                    shard.observation_count
                )
            )

            number_batches = int(
                math.ceil(
                    shard.observation_count
                    /
                    BATCH_SIZE
                )
            )

            start_batch = (
                batch_position
                if position == shard_position
                else 0
            )

            for batch_index in range(
                start_batch,
                number_batches,
            ):
                start = (
                    batch_index
                    *
                    BATCH_SIZE
                )

                stop = min(
                    start
                    +
                    BATCH_SIZE,
                    shard.observation_count,
                )

                batch_indices = (
                    observation_order[
                        start:
                        stop
                    ]
                )

                loss_sum, count = (
                    adb_train_batch(
                        model=model,
                        optimizer=optimizer,
                        cache_arrays=cache_arrays,
                        observations=observations,
                        observation_indices=batch_indices,
                        device=device,
                    )
                )

                epoch_loss_sum += float(
                    loss_sum
                )

                epoch_count += int(
                    count
                )

                global_steps += 1

                if (
                    global_steps
                    %
                    1000
                    ==
                    0
                ):
                    save_adb_resume(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        shard_position=position,
                        batch_position=batch_index + 1,
                        epoch_loss_sum=epoch_loss_sum,
                        epoch_count=epoch_count,
                        global_steps=global_steps,
                        best_loss=best_loss,
                        best_epoch=best_epoch,
                        patience=patience,
                        history=history,
                    )

                    print(
                        (
                            f"ADB epoch {epoch:02d} | "
                            f"step={global_steps} | "
                            f"shard={position + 1}/"
                            f"{len(fit_shards)} | "
                            f"obs={epoch_count}"
                        ),
                        flush=True,
                    )

            # Finished this shard.
            batch_position = 0

            save_adb_resume(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                shard_position=position + 1,
                batch_position=0,
                epoch_loss_sum=epoch_loss_sum,
                epoch_count=epoch_count,
                global_steps=global_steps,
                best_loss=best_loss,
                best_epoch=best_epoch,
                patience=patience,
                history=history,
            )

            if (
                (
                    position + 1
                )
                %
                100
                ==
                0
                or
                (
                    position + 1
                )
                ==
                len(
                    fit_shards
                )
            ):
                print(
                    (
                        f"ADB epoch {epoch:02d} | "
                        f"shards={position + 1}/"
                        f"{len(fit_shards)} | "
                        f"obs={epoch_count}"
                    ),
                    flush=True,
                )

        require(
            epoch_count > 0,
            "ADB epoch has zero observations.",
        )

        train_loss = (
            epoch_loss_sum
            /
            epoch_count
        )

        dev_loss = evaluate_adb(
            model,
            dev_shards,
            normalizer,
            device,
        )

        require(
            math.isfinite(
                dev_loss
            ),
            "ADB development BCE non-finite.",
        )

        improved = (
            dev_loss
            <
            best_loss
        )

        if improved:
            best_loss = float(
                dev_loss
            )

            best_epoch = int(
                epoch
            )

            patience = 0

            save_best(
                path=ADB_BEST,
                task="direct_ADB",
                model=model,
                epoch=epoch,
                dev_loss=dev_loss,
            )

        else:
            patience += 1

        history.append(
            {
                "epoch":
                    int(epoch),

                "train_loss":
                    float(
                        train_loss
                    ),

                "development_loss":
                    float(
                        dev_loss
                    ),
            }
        )

        print(
            (
                f"ADB epoch {epoch:02d} | "
                f"trainBCE={train_loss:.9f} | "
                f"devBCE={dev_loss:.9f} | "
                f"best={best_loss:.9f} | "
                f"bestEpoch={best_epoch} | "
                f"patience={patience}/{PATIENCE}"
            ),
            flush=True,
        )

        if patience >= PATIENCE:
            print(
                "ADB early stopping = TRIGGERED"
            )
            break

        epoch += 1
        shard_position = 0
        batch_position = 0
        epoch_loss_sum = 0.0
        epoch_count = 0

        save_adb_resume(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            shard_position=0,
            batch_position=0,
            epoch_loss_sum=0.0,
            epoch_count=0,
            global_steps=global_steps,
            best_loss=best_loss,
            best_epoch=best_epoch,
            patience=patience,
            history=history,
        )

    require(
        global_steps > 0,
        "ADB optimizer never stepped.",
    )

    best = load_model_checkpoint(
        model,
        ADB_BEST,
        device,
    )

    final_best_loss = evaluate_adb(
        model,
        dev_shards,
        normalizer,
        device,
    )

    require(
        abs(
            final_best_loss
            -
            float(
                best[
                    "development_task_loss"
                ]
            )
        )
        <=
        1.0e-10,
        "ADB best-checkpoint dev loss mismatch.",
    )

    return {
        "seed":
            ADB_SEED,

        "fit_shards":
            len(
                fit_shards
            ),

        "development_shards":
            len(
                dev_shards
            ),

        "fit_observations":
            int(
                sum(
                    shard.observation_count
                    for shard in fit_shards
                )
            ),

        "development_observations":
            int(
                sum(
                    shard.observation_count
                    for shard in dev_shards
                )
            ),

        "initial_development_loss":
            float(
                history[0][
                    "development_loss"
                ]
            ),

        "best_epoch":
            int(
                best[
                    "epoch"
                ]
            ),

        "best_development_loss":
            float(
                best[
                    "development_task_loss"
                ]
            ),

        "optimizer_steps":
            int(
                global_steps
            ),

        "history":
            history,

        "trainable_parameter_count":
            parameter_count(
                model
            ),

        "checkpoint_path":
            str(
                ADB_BEST
            ),

        "checkpoint_sha256":
            sha256_file(
                ADB_BEST
            ),

        "state_dict_sha256":
            best[
                "state_dict_sha256"
            ],
    }


# ============================================================
# Probe mode for fresh-process repeat
# ============================================================

def probe(
    task,
    checkpoint_path,
):
    for path, expected in EXPECTED.items():
        require(
            sha256_file(
                path
            )
            ==
            expected,
            f"Probe authority changed: {path}",
        )

    normalizer = (
        FitInputNormalizer(
            NORMALIZATION
        )
    )

    device = choose_device()

    if task == "beam":
        set_global_determinism(
            BEAM_SEED
        )

        model = (
            DirectBeamClassifier()
            .to(
                device
            )
        )

        load_model_checkpoint(
            model,
            checkpoint_path,
            device,
        )

        arrays = build_beam_arrays(
            "development",
            normalizer,
        )

        count = min(
            8,
            int(
                arrays.target.shape[0]
            ),
        )

        batch = beam_batch_to_device(
            arrays,
            np.arange(
                count,
                dtype=np.int64,
            ),
            device,
        )

        model.eval()

        with torch.no_grad():
            outputs = model(
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

        digest = tensor_sha256(
            [
                (
                    f"beam_{size}",
                    outputs[size],
                )
                for size in CODEBOOKS
            ]
        )

    elif task == "adb":
        set_global_determinism(
            ADB_SEED
        )

        model = (
            DirectADBField()
            .to(
                device
            )
        )

        load_model_checkpoint(
            model,
            checkpoint_path,
            device,
        )

        shards = scan_adb_partition(
            "development"
        )

        shard = next(
            item
            for item in shards
            if item.observation_count
            >
            0
        )

        cache_arrays = load_adb_cache(
            shard,
            normalizer,
        )

        observations = (
            load_adb_observations(
                shard
            )
        )

        (
            target_all,
            neighbors_all,
            neighbor_mask_all,
            map_all,
            class_all,
        ) = cache_arrays

        model.eval()

        with torch.no_grad():
            embedding_all = model.encode(
                torch.from_numpy(
                    target_all
                ).to(
                    device
                ),
                torch.from_numpy(
                    neighbors_all
                ).to(
                    device
                ),
                torch.from_numpy(
                    neighbor_mask_all
                ).to(
                    device
                ),
                torch.from_numpy(
                    map_all
                ).to(
                    device
                ),
            )

            count = min(
                512,
                shard.observation_count,
            )

            idx = np.arange(
                count,
                dtype=np.int64,
            )

            rows_np = observations[
                "row"
            ][
                idx
            ]

            rows = torch.from_numpy(
                rows_np
            ).to(
                device=device,
                dtype=torch.long,
            )

            logits = model.decode(
                embedding_all[
                    rows
                ],
                torch.from_numpy(
                    observations[
                        "theta"
                    ][
                        idx
                    ]
                ).to(
                    device=device,
                    dtype=torch.long,
                ),
                torch.from_numpy(
                    observations[
                        "range"
                    ][
                        idx
                    ]
                ).to(
                    device=device,
                    dtype=torch.long,
                ),
                torch.from_numpy(
                    observations[
                        "horizon"
                    ][
                        idx
                    ]
                ).to(
                    device=device,
                    dtype=torch.long,
                ),
                torch.from_numpy(
                    class_all[
                        rows_np
                    ]
                ).to(
                    device=device,
                    dtype=torch.long,
                ),
            )

        digest = tensor_sha256(
            [
                (
                    "adb_logits",
                    logits,
                )
            ]
        )

    else:
        raise FailClosed(
            f"Unknown probe task: {task}"
        )

    print(
        "PROBE_JSON="
        +
        json.dumps(
            {
                "task":
                    task,

                "checkpoint_sha256":
                    sha256_file(
                        checkpoint_path
                    ),

                "output_sha256":
                    digest,

                "formal_N120_raw_data_opened":
                    False,
            },
            sort_keys=True,
        )
    )


# ============================================================
# Main training
# ============================================================

def main():
    print("=" * 78)
    print("STAGE 7 — BLOCK 7.4H-D")
    print("DIRECT BEAM + DIRECT ADB BASELINE TRAINING")
    print("FIRST AUTHORIZED DIRECT OPTIMIZER EXECUTION")
    print("FORMAL N120 RAW DATA SEALED")
    print("=" * 78)

    for path, expected in EXPECTED.items():
        require(
            path.is_file(),
            f"Missing authority: {path}",
        )

        actual = sha256_file(
            path
        )

        require(
            actual == expected,
            (
                f"Authority changed:\n"
                f"{path}\n"
                f"expected={expected}\n"
                f"actual={actual}"
            ),
        )

        print(
            "PASS",
            path,
            actual,
        )

    prereg = load_json(
        PREREG
    )

    require(
        prereg.get(
            "status"
        )
        ==
        "FROZEN_PREOPTIMIZER_DIRECT_BASELINE_TRAINING_PREREGISTRATION",
        "H-C preregistration status changed.",
    )

    require(
        prereg[
            "common_encoder"
        ][
            "initialization"
        ]
        ==
        "fresh_seeded_no_weight_transfer",
        "Direct initialization changed.",
    )

    require(
        prereg[
            "common_encoder"
        ][
            "Stage4_predictive_weights_copied"
        ]
        is False,
        "Stage4 weight transfer not allowed.",
    )

    require(
        prereg[
            "model_selection"
        ][
            "minimum_delta"
        ]
        ==
        0.0,
        "Minimum delta changed.",
    )

    # A finalized report makes retraining forbidden.
    if REPORT.is_file():
        existing = load_json(
            REPORT
        )

        if (
            existing.get(
                "status"
            )
            ==
            "PASS_DIRECT_BASELINE_TRAINING_COMPLETE"
        ):
            print()
            print(
                "Existing finalized H-D report found."
            )
            print(
                "RETRAINING = FORBIDDEN"
            )
            print(
                "Exact readback only."
            )
            print(
                "report SHA256 =",
                sha256_file(
                    REPORT
                ),
            )
            return

        raise FailClosed(
            (
                "Non-final existing H-D report "
                "requires explicit provenance repair; "
                "refusing silent restart."
            )
        )

    device = choose_device()

    print()
    print(
        "device =",
        str(
            device
        ),
    )

    print(
        "torch version =",
        torch.__version__,
    )

    print(
        "cuda runtime =",
        torch.version.cuda,
    )

    if (
        device.type
        ==
        "cuda"
    ):
        print(
            "GPU =",
            torch.cuda.get_device_name(
                0
            ),
        )

    normalizer = (
        FitInputNormalizer(
            NORMALIZATION
        )
    )

    print()
    print("===== A. LOAD BEAM CORPUS =====")

    beam_fit = build_beam_arrays(
        "fit",
        normalizer,
    )

    beam_dev = build_beam_arrays(
        "development",
        normalizer,
    )

    require(
        (
            beam_fit.target.shape[0]
            +
            beam_dev.target.shape[0]
        )
        ==
        EXPECTED_BEAM_TOTAL,
        "Beam population changed.",
    )

    print(
        "beam fit examples =",
        beam_fit.target.shape[0],
    )

    print(
        "beam dev examples =",
        beam_dev.target.shape[0],
    )

    print()
    print("===== B. SCAN ADB CORPUS =====")

    adb_fit = scan_adb_partition(
        "fit"
    )

    adb_dev = scan_adb_partition(
        "development"
    )

    adb_total = int(
        sum(
            shard.observation_count
            for shard in
            (
                adb_fit
                +
                adb_dev
            )
        )
    )

    adb_positive = int(
        sum(
            shard.positive_count
            for shard in
            (
                adb_fit
                +
                adb_dev
            )
        )
    )

    adb_negative = int(
        sum(
            shard.negative_count
            for shard in
            (
                adb_fit
                +
                adb_dev
            )
        )
    )

    require(
        adb_total
        ==
        EXPECTED_ADB_TOTAL,
        (
            f"ADB total observations "
            f"changed: {adb_total}"
        ),
    )

    require(
        adb_positive
        ==
        EXPECTED_ADB_POSITIVE,
        "ADB positive count changed.",
    )

    require(
        adb_negative
        ==
        EXPECTED_ADB_NEGATIVE,
        "ADB negative count changed.",
    )

    print(
        "ADB total observations =",
        adb_total,
    )

    print(
        "ADB positive =",
        adb_positive,
    )

    print(
        "ADB negative =",
        adb_negative,
    )

    # --------------------------------------------------------
    # Actual first optimizer execution begins below.
    # No configuration mutation is allowed after this point.
    # --------------------------------------------------------

    beam_result = train_beam(
        fit=beam_fit,
        dev=beam_dev,
        device=device,
    )

    adb_result = train_adb(
        fit_shards=adb_fit,
        dev_shards=adb_dev,
        normalizer=normalizer,
        device=device,
    )

    require(
        beam_result[
            "optimizer_steps"
        ]
        >
        0,
        "Beam optimizer-step count is zero.",
    )

    require(
        adb_result[
            "optimizer_steps"
        ]
        >
        0,
        "ADB optimizer-step count is zero.",
    )

    result = {
        "stage":
            7,

        "block":
            "7.4H-D",

        "status":
            "PASS_DIRECT_BASELINE_TRAINING_COMPLETE",

        "trainer_sha256":
            sha256_file(
                Path(
                    __file__
                )
            ),

        "preregistration": {
            "path":
                str(
                    PREREG
                ),

            "sha256":
                sha256_file(
                    PREREG
                ),
        },

        "device": {
            "torch":
                torch.__version__,

            "device":
                str(
                    device
                ),

            "cuda_runtime":
                torch.version.cuda,

            "gpu_name":
                (
                    torch.cuda
                    .get_device_name(
                        0
                    )
                    if (
                        device.type
                        ==
                        "cuda"
                    )
                    else None
                ),
        },

        "optimizer_envelope": {
            "optimizer":
                "AdamW",

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "batch_size":
                BATCH_SIZE,

            "max_epochs":
                MAX_EPOCHS,

            "early_stopping_patience":
                PATIENCE,

            "minimum_delta":
                0.0,

            "gradient_clip_norm":
                GRADIENT_CLIP,

            "mixed_precision":
                False,
        },

        "direct_beam":
            beam_result,

        "direct_ADB":
            adb_result,

        "scientific_boundary": {
            "training_population":
                "fit only",

            "model_selection_population":
                "development only",

            "calibration_partition_opened":
                False,

            "formal_N120_raw_data_opened":
                False,

            "formal_outcomes_opened":
                False,

            "future_GT_as_inference_input":
                False,

            "Stage4_modified":
                False,

            "Stage5_modified":
                False,

            "Stage6_modified":
                False,

            "label_shards_modified":
                False,

            "post_outcome_hyperparameter_change":
                False,

            "post_outcome_retraining":
                False,
        },

        "next":
            (
                "independent H-D checkpoint/"
                "fresh-process repeat checker"
            ),
    }

    atomic_json(
        REPORT,
        result,
    )

    print()
    print("=" * 78)
    print("BLOCK 7.4H-D TRAINING = COMPLETE")
    print(
        "beam best epoch =",
        beam_result[
            "best_epoch"
        ],
    )
    print(
        "beam best dev CE =",
        beam_result[
            "best_development_loss"
        ],
    )
    print(
        "ADB best epoch =",
        adb_result[
            "best_epoch"
        ],
    )
    print(
        "ADB best dev BCE =",
        adb_result[
            "best_development_loss"
        ],
    )
    print("FORMAL N120 RAW DATA = SEALED")
    print("CALIBRATION PARTITION = UNOPENED")
    print("POST-OUTCOME RETUNING = FORBIDDEN")
    print("NEXT = INDEPENDENT H-D CHECKER")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--probe",
        choices=(
            "beam",
            "adb",
        ),
        default=None,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    if args.probe is not None:
        require(
            args.checkpoint is not None,
            "--checkpoint required for probe.",
        )

        probe(
            args.probe,
            args.checkpoint,
        )

    else:
        main()

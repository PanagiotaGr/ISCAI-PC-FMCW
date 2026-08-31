from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch


FORMAL_ARRAY_KEYS = (
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


def canonical_json_sha256(
    payload,
):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=False,
        allow_nan=False,
    ).encode(
        "utf-8"
    )

    return sha256(
        encoded
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


def checkpoint_state_sha256(
    path,
):
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        "state_dict"
        not in payload
    ):
        raise RuntimeError(
            f"{path} has no state_dict."
        )

    return state_dict_sha256(
        payload[
            "state_dict"
        ]
    )


def formal_cache_content_sha256(
    formal_rows,
    cache_dir,
):
    """
    Reconstruct exactly the Block4.8
    merged-content digest:

      all merged arrays in frozen key order
      +
      per-scenario all-prediction hashes
        in formal-manifest order.

    This intentionally does NOT trust the
    merged .npz file or Block4.8 report.
    """

    cache_dir = Path(
        cache_dir
    )

    parts = {
        key:
            []
        for key in FORMAL_ARRAY_KEYS
    }

    prediction_hashes = []

    cache_contracts = set()

    matched_total = 0
    eligible_total = 0

    shard_files = []

    for rank, row in enumerate(
        formal_rows,
        start=1,
    ):
        scenario_id = str(
            row[
                "scenario_id"
            ]
        )

        path = (
            cache_dir
            /
            (
                f"{rank:03d}_"
                f"{scenario_id}.npz"
            )
        )

        if not path.is_file():
            raise RuntimeError(
                "Missing formal cache shard: "
                f"{path}"
            )

        shard_files.append(
            path
        )

        with np.load(
            path,
            allow_pickle=False,
        ) as data:

            stored_scenario_id = str(
                data[
                    "scenario_id"
                ].item()
            )

            if (
                stored_scenario_id
                !=
                scenario_id
            ):
                raise RuntimeError(
                    "Formal cache scenario-ID "
                    f"mismatch in {path.name}."
                )

            if not bool(
                data[
                    "tracks_to_predict_accessed_after_prediction"
                ].item()
            ):
                raise RuntimeError(
                    "Formal cache leakage contract "
                    f"failed in {path.name}."
                )

            cache_contracts.add(
                str(
                    data[
                        "cache_contract_sha256"
                    ].item()
                )
            )

            for key in (
                FORMAL_ARRAY_KEYS
            ):
                if key not in data:
                    raise RuntimeError(
                        f"{path.name} lacks {key}."
                    )

                value = np.asarray(
                    data[
                        key
                    ]
                )

                if (
                    np.issubdtype(
                        value.dtype,
                        np.number,
                    )
                    and
                    not np.isfinite(
                        value
                    ).all()
                ):
                    raise RuntimeError(
                        "Non-finite formal-cache "
                        f"array {path.name}:{key}."
                    )

                parts[
                    key
                ].append(
                    value
                )

            prediction_sha = str(
                data[
                    "all_prediction_sha256"
                ].item()
            )

            if (
                len(
                    prediction_sha
                )
                !=
                64
            ):
                raise RuntimeError(
                    "Invalid per-scene prediction "
                    f"SHA in {path.name}."
                )

            prediction_hashes.append(
                prediction_sha
            )

            matched_total += int(
                data[
                    "matched_formal_target_count"
                ].item()
            )

            eligible_total += int(
                data[
                    "eligible_formal_target_count"
                ].item()
            )

    if (
        len(
            cache_contracts
        )
        !=
        1
    ):
        raise RuntimeError(
            "Formal shards do not share "
            "one cache-contract SHA."
        )

    merged = {}

    for key in (
        FORMAL_ARRAY_KEYS
    ):
        merged[
            key
        ] = np.concatenate(
            parts[
                key
            ],
            axis=0,
        )

    digest = sha256()

    for key in (
        FORMAL_ARRAY_KEYS
    ):
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

    for prediction_sha in (
        prediction_hashes
    ):
        digest.update(
            prediction_sha.encode(
                "ascii"
            )
        )

    return {
        "sha256":
            digest.hexdigest(),

        "shard_count":
            len(
                shard_files
            ),

        "cache_contract_sha256":
            next(
                iter(
                    cache_contracts
                )
            ),

        "matched_formal_targets":
            int(
                matched_total
            ),

        "eligible_formal_targets":
            int(
                eligible_total
            ),

        "merged_shapes": {
            key:
                list(
                    merged[
                        key
                    ].shape
                )
            for key in (
                FORMAL_ARRAY_KEYS
            )
        },

        "per_scene_prediction_hash_count":
            len(
                prediction_hashes
            ),
    }

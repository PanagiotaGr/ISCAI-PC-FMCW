from __future__ import annotations

from collections import Counter
from dataclasses import fields, is_dataclass
from hashlib import sha256
import importlib
import inspect
import math

import numpy as np
import torch


HORIZONS_S = (
    0.1,
    0.3,
    0.5,
    1.0,
)

HORIZON_NAMES = (
    "0.1",
    "0.3",
    "0.5",
    "1.0",
)

HORIZON_FRAME_OFFSETS = (
    1,
    3,
    5,
    10,
)


def public_field_names(
    value,
):
    names = []

    if is_dataclass(
        value
    ):
        names.extend(
            field.name
            for field in fields(
                value
            )
        )

    if hasattr(
        value,
        "__dict__",
    ):
        names.extend(
            key
            for key in value.__dict__
            if not key.startswith("_")
        )

    if not names:
        names.extend(
            name
            for name in dir(
                value
            )
            if not name.startswith("_")
        )

    return tuple(
        sorted(
            set(
                names
            )
        )
    )


def resolve_sample_truth_track_index(
    sample,
):
    """
    Resolve the explicit evaluator metadata
    produced by the frozen Block4.2 supervision
    pairing.

    Never use prediction_id as truth identity.
    """

    candidate_names = (
        "truth_track_index",
        "truth_index",
        "matched_truth_track_index",
        "womd_track_index",
        "ground_truth_track_index",
        "target_track_index",
    )

    objects = [
        (
            "sample",
            sample,
        )
    ]

    for nested_name in (
        "future_label",
        "supervision",
        "metadata",
    ):
        nested = getattr(
            sample,
            nested_name,
            None,
        )

        if nested is not None:
            objects.append(
                (
                    nested_name,
                    nested,
                )
            )

    resolved = []

    for origin, value in objects:
        for name in candidate_names:
            if not hasattr(
                value,
                name,
            ):
                continue

            item = getattr(
                value,
                name,
            )

            if isinstance(
                item,
                np.integer,
            ):
                item = int(
                    item
                )

            if isinstance(
                item,
                int,
            ):
                resolved.append(
                    (
                        f"{origin}.{name}",
                        item,
                    )
                )

    if not resolved:
        evidence = {
            origin:
                public_field_names(
                    value
                )
            for origin, value
            in objects
        }

        raise RuntimeError(
            "Could not resolve explicit "
            "truth-track index from supervised "
            f"sample. Fields={evidence}"
        )

    values = {
        item
        for _, item
        in resolved
    }

    if len(values) != 1:
        raise RuntimeError(
            "Conflicting truth-track index "
            f"metadata: {resolved}"
        )

    return {
        "index":
            resolved[
                0
            ][
                1
            ],

        "source":
            resolved[
                0
            ][
                0
            ],

        "all_matches":
            tuple(
                resolved
            ),
    }


def tracks_to_predict_indices(
    scenario,
):
    values = []

    for item in (
        scenario.tracks_to_predict
    ):
        if not hasattr(
            item,
            "track_index",
        ):
            raise RuntimeError(
                "tracks_to_predict element "
                "lacks track_index."
            )

        index = int(
            item.track_index
        )

        if (
            index < 0
            or
            index
            >=
            len(
                scenario.tracks
            )
        ):
            raise RuntimeError(
                "tracks_to_predict contains "
                f"invalid track index {index}."
            )

        values.append(
            index
        )

    if (
        len(
            set(
                values
            )
        )
        !=
        len(
            values
        )
    ):
        raise RuntimeError(
            "tracks_to_predict contains "
            "duplicate track indices."
        )

    return tuple(
        values
    )


def eligible_formal_targets(
    scenario,
):
    """
    Must be called only after prediction generation.

    Returns target-level evaluator metadata and
    future-validity at the four frozen horizons.
    """

    indices = tracks_to_predict_indices(
        scenario
    )

    current_index = int(
        scenario.current_time_index
    )

    if current_index != 10:
        raise RuntimeError(
            "Frozen WOMD current_time_index "
            f"changed: {current_index}"
        )

    classes = []
    validity = []

    for index in indices:
        track = scenario.tracks[
            index
        ]

        classes.append(
            int(
                track.object_type
            )
        )

        row = []

        for offset in (
            HORIZON_FRAME_OFFSETS
        ):
            state_index = (
                current_index
                +
                offset
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

            row.append(
                bool(
                    valid
                )
            )

        validity.append(
            row
        )

    return {
        "track_index":
            np.asarray(
                indices,
                dtype=np.int64,
            ),

        "class_id":
            np.asarray(
                classes,
                dtype=np.int64,
            ),

        "validity":
            np.asarray(
                validity,
                dtype=np.bool_,
            ).reshape(
                len(
                    indices
                ),
                4,
            ),
    }


def construct_deterministic_model(
    architecture_configuration,
    state_dict,
    *,
    device,
):
    """
    Discover the frozen deterministic class
    rather than assuming a class name.
    """

    section = (
        architecture_configuration[
            "model"
        ]
    )

    candidate_modules = (
        "iscai_stage4.ml.deterministic_gru",
        "iscai_stage4.ml",
    )

    candidate_names = (
        "DeterministicTrajectoryGRU",
        "TrajectoryGRU",
        "DeterministicGRU",
    )

    attempts = []

    for module_name in (
        candidate_modules
    ):
        try:
            module = importlib.import_module(
                module_name
            )

        except Exception as exc:
            attempts.append(
                (
                    module_name,
                    "<module>",
                    f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        for class_name in (
            candidate_names
        ):
            cls = getattr(
                module,
                class_name,
                None,
            )

            if (
                cls is None
                or
                not inspect.isclass(
                    cls
                )
            ):
                continue

            try:
                signature = inspect.signature(
                    cls
                )

                available = {
                    "target_hidden_dim":
                        int(
                            section[
                                "target_hidden_dim"
                            ]
                        ),

                    "neighbor_hidden_dim":
                        int(
                            section[
                                "neighbor_hidden_dim"
                            ]
                        ),

                    "map_hidden_dim":
                        int(
                            section[
                                "map_hidden_dim"
                            ]
                        ),

                    "fusion_hidden_dim":
                        int(
                            section[
                                "fusion_hidden_dim"
                            ]
                        ),

                    "use_neighbors":
                        True,

                    "use_map":
                        True,
                }

                kwargs = {
                    name:
                        value
                    for name, value
                    in available.items()
                    if name
                    in
                    signature.parameters
                }

                model = cls(
                    **kwargs
                ).to(
                    device
                )

                model.load_state_dict(
                    state_dict,
                    strict=True,
                )

                model.eval()

                return (
                    model,
                    (
                        f"{module_name}."
                        f"{class_name}"
                    ),
                )

            except Exception as exc:
                attempts.append(
                    (
                        module_name,
                        class_name,
                        f"{type(exc).__name__}: {exc}",
                    )
                )

    raise RuntimeError(
        "Could not reconstruct frozen "
        "deterministic GRU. Attempts="
        +
        repr(
            attempts
        )
    )


def extract_deterministic_prediction(
    output,
):
    if isinstance(
        output,
        torch.Tensor,
    ):
        candidate = output

    else:
        candidate = None

        for name in (
            "prediction",
            "mean",
            "trajectory",
            "positions",
        ):
            value = getattr(
                output,
                name,
                None,
            )

            if isinstance(
                value,
                torch.Tensor,
            ):
                candidate = value
                break

        if (
            candidate is None
            and
            isinstance(
                output,
                (
                    tuple,
                    list,
                ),
            )
            and
            output
            and
            isinstance(
                output[
                    0
                ],
                torch.Tensor,
            )
        ):
            candidate = output[
                0
            ]

    if candidate is None:
        raise RuntimeError(
            "Could not extract deterministic "
            "trajectory tensor from model output."
        )

    if (
        candidate.ndim != 3
        or
        tuple(
            candidate.shape[
                1:
            ]
        )
        !=
        (
            4,
            3,
        )
    ):
        raise RuntimeError(
            "Deterministic prediction shape "
            f"changed: {tuple(candidate.shape)}"
        )

    return candidate


def tensor_bundle_sha256(
    tensors,
):
    digest = sha256()

    for name, tensor in tensors:
        digest.update(
            name.encode(
                "utf-8"
            )
        )
        digest.update(
            b"\0"
        )

        array = (
            tensor.detach()
            .cpu()
            .contiguous()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        digest.update(
            array.tobytes(
                order="C"
            )
        )
        digest.update(
            b"\0"
        )

    return digest.hexdigest()


def trajectory_point_metrics(
    mean,
    truth,
    validity,
    class_id,
    *,
    class_id_to_name,
):
    mean = np.asarray(
        mean,
        dtype=np.float64,
    )

    truth = np.asarray(
        truth,
        dtype=np.float64,
    )

    valid = np.asarray(
        validity,
        dtype=np.bool_,
    )

    classes = np.asarray(
        class_id,
        dtype=np.int64,
    )

    if (
        mean.shape != truth.shape
        or
        mean.ndim != 3
        or
        mean.shape[
            1:
        ] != (
            4,
            3,
        )
        or
        valid.shape
        !=
        mean.shape[
            :2
        ]
        or
        classes.shape
        !=
        (
            mean.shape[
                0
            ],
        )
    ):
        raise ValueError(
            "Trajectory metric shape mismatch."
        )

    planar = np.linalg.norm(
        mean[
            ...,
            :2
        ]
        -
        truth[
            ...,
            :2
        ],
        axis=-1,
    )

    error_3d = np.linalg.norm(
        mean
        -
        truth,
        axis=-1,
    )

    if not np.any(
        valid
    ):
        raise ValueError(
            "No valid matched actor-horizon points."
        )

    horizon = {}

    for index, name in enumerate(
        HORIZON_NAMES
    ):
        mask = valid[
            :,
            index
        ]

        horizon[
            name
        ] = {
            "available_predictions":
                int(
                    mask.sum()
                ),

            "mean_planar_error_m":
                float(
                    np.mean(
                        planar[
                            mask,
                            index
                        ]
                    )
                )
                if np.any(
                    mask
                )
                else None,

            "mean_3d_error_m":
                float(
                    np.mean(
                        error_3d[
                            mask,
                            index
                        ]
                    )
                )
                if np.any(
                    mask
                )
                else None,
        }

    final_valid = valid[
        :,
        3
    ]

    classes_report = {}

    for value in sorted(
        set(
            classes.tolist()
        )
    ):
        sample_mask = (
            classes
            ==
            value
        )

        point_mask = (
            sample_mask[
                :,
                None
            ]
            &
            valid
        )

        if not np.any(
            point_mask
        ):
            continue

        name = class_id_to_name.get(
            int(
                value
            ),
            str(
                int(
                    value
                )
            ),
        )

        class_horizons = {}

        for h, horizon_name in enumerate(
            HORIZON_NAMES
        ):
            hmask = (
                sample_mask
                &
                valid[
                    :,
                    h
                ]
            )

            class_horizons[
                horizon_name
            ] = {
                "available_predictions":
                    int(
                        hmask.sum()
                    ),

                "mean_planar_error_m":
                    float(
                        np.mean(
                            planar[
                                hmask,
                                h
                            ]
                        )
                    )
                    if np.any(
                        hmask
                    )
                    else None,
            }

        class_final = (
            sample_mask
            &
            final_valid
        )

        classes_report[
            name
        ] = {
            "matched_tracks":
                int(
                    sample_mask.sum()
                ),

            "ade_m":
                float(
                    np.mean(
                        planar[
                            point_mask
                        ]
                    )
                ),

            "ade_3d_m":
                float(
                    np.mean(
                        error_3d[
                            point_mask
                        ]
                    )
                ),

            "fde_1.0s_m":
                float(
                    np.mean(
                        planar[
                            class_final,
                            3
                        ]
                    )
                )
                if np.any(
                    class_final
                )
                else None,

            "horizons":
                class_horizons,
        }

    return {
        "ade_m":
            float(
                np.mean(
                    planar[
                        valid
                    ]
                )
            ),

        "ade_3d_m":
            float(
                np.mean(
                    error_3d[
                        valid
                    ]
                )
            ),

        "ade_point_count":
            int(
                valid.sum()
            ),

        "matched_track_count":
            int(
                mean.shape[
                    0
                ]
            ),

        "fde_1.0s_m":
            float(
                np.mean(
                    planar[
                        final_valid,
                        3
                    ]
                )
            )
            if np.any(
                final_valid
            )
            else None,

        "fde_1.0s_track_count":
            int(
                final_valid.sum()
            ),

        "horizons":
            horizon,

        "classes":
            classes_report,
    }


def target_availability_metrics(
    *,
    matched_truth_index,
    matched_class_id,
    matched_validity,
    eligible_truth_index,
    eligible_class_id,
    eligible_validity,
    class_id_to_name,
):
    matched_truth_index = np.asarray(
        matched_truth_index,
        dtype=np.int64,
    )

    matched_class_id = np.asarray(
        matched_class_id,
        dtype=np.int64,
    )

    matched_validity = np.asarray(
        matched_validity,
        dtype=np.bool_,
    )

    eligible_truth_index = np.asarray(
        eligible_truth_index,
        dtype=np.int64,
    )

    eligible_class_id = np.asarray(
        eligible_class_id,
        dtype=np.int64,
    )

    eligible_validity = np.asarray(
        eligible_validity,
        dtype=np.bool_,
    )

    if len(
        np.unique(
            matched_truth_index
        )
    ) != len(
        matched_truth_index
    ):
        raise ValueError(
            "Matched truth indices must be unique "
            "within the aggregation namespace."
        )

    matched_count = int(
        len(
            matched_truth_index
        )
    )

    eligible_count = int(
        len(
            eligible_truth_index
        )
    )

    report = {
        "matched_tracks":
            matched_count,

        "eligible_truth_tracks":
            eligible_count,

        "matched_target_recall":
            (
                float(
                    matched_count
                    /
                    eligible_count
                )
                if eligible_count
                else None
            ),

        "horizons":
            {},

        "classes":
            {},
    }

    for h, name in enumerate(
        HORIZON_NAMES
    ):
        available = int(
            matched_validity[
                :,
                h
            ].sum()
        )

        eligible = int(
            eligible_validity[
                :,
                h
            ].sum()
        )

        report[
            "horizons"
        ][
            name
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

            "unavailable_prediction_rate":
                (
                    float(
                        1.0
                        -
                        available
                        /
                        eligible
                    )
                    if eligible
                    else None
                ),
        }

    all_classes = sorted(
        set(
            eligible_class_id.tolist()
        )
        |
        set(
            matched_class_id.tolist()
        )
    )

    for class_value in all_classes:
        class_value = int(
            class_value
        )

        name = class_id_to_name.get(
            class_value,
            str(
                class_value
            ),
        )

        eligible_mask = (
            eligible_class_id
            ==
            class_value
        )

        matched_mask = (
            matched_class_id
            ==
            class_value
        )

        eligible = int(
            eligible_mask.sum()
        )

        matched = int(
            matched_mask.sum()
        )

        report[
            "classes"
        ][
            name
        ] = {
            "eligible_truth_tracks":
                eligible,

            "matched_tracks":
                matched,

            "recall":
                (
                    float(
                        matched
                        /
                        eligible
                    )
                    if eligible
                    else None
                ),
        }

    return report


def gmm_formal_metrics(
    mixture_logits,
    means,
    truth,
    validity,
):
    logits = np.asarray(
        mixture_logits,
        dtype=np.float64,
    )

    means = np.asarray(
        means,
        dtype=np.float64,
    )

    truth = np.asarray(
        truth,
        dtype=np.float64,
    )

    valid = np.asarray(
        validity,
        dtype=np.bool_,
    )

    if (
        logits.ndim != 2
        or
        logits.shape[
            1
        ] != 3
        or
        means.shape
        !=
        (
            logits.shape[
                0
            ],
            3,
            4,
            3,
        )
        or
        truth.shape
        !=
        (
            logits.shape[
                0
            ],
            4,
            3,
        )
        or
        valid.shape
        !=
        (
            logits.shape[
                0
            ],
            4,
        )
    ):
        raise ValueError(
            "GMM formal metric shape mismatch."
        )

    shifted = (
        logits
        -
        np.max(
            logits,
            axis=1,
            keepdims=True,
        )
    )

    probabilities = np.exp(
        shifted
    )

    probabilities = (
        probabilities
        /
        probabilities.sum(
            axis=1,
            keepdims=True,
        )
    )

    mixture_mean = np.sum(
        probabilities[
            :,
            :,
            None,
            None
        ]
        *
        means,
        axis=1,
    )

    map_index = np.argmax(
        logits,
        axis=1,
    )

    map_mean = means[
        np.arange(
            len(
                means
            )
        ),
        map_index,
    ]

    component_planar = np.linalg.norm(
        means[
            ...,
            :2
        ]
        -
        truth[
            :,
            None,
            :,
            :2
        ],
        axis=-1,
    )

    active = valid.any(
        axis=1
    )

    denominator = np.maximum(
        valid.sum(
            axis=1
        ),
        1,
    )

    component_ade = (
        (
            component_planar
            *
            valid[
                :,
                None,
                :
            ]
        ).sum(
            axis=2
        )
        /
        denominator[
            :,
            None
        ]
    )

    minade = np.min(
        component_ade,
        axis=1,
    )

    final_valid = valid[
        :,
        3
    ]

    minfde = np.min(
        component_planar[
            :,
            :,
            3
        ],
        axis=1,
    )

    p = np.maximum(
        probabilities,
        1e-12,
    )

    entropy = -np.sum(
        p
        *
        np.log(
            p
        ),
        axis=1,
    )

    map_counts = Counter(
        int(
            item
        )
        for item in (
            map_index.tolist()
        )
    )

    return {
        "mixture_mean":
            mixture_mean,

        "MAP_mean":
            map_mean,

        "minADE_m":
            float(
                np.mean(
                    minade[
                        active
                    ]
                )
            ),

        "minFDE_1.0s_m":
            float(
                np.mean(
                    minfde[
                        final_valid
                    ]
                )
            )
            if np.any(
                final_valid
            )
            else None,

        "mixture_entropy_nats":
            float(
                np.mean(
                    entropy[
                        active
                    ]
                )
            ),

        "effective_mode_count":
            float(
                np.mean(
                    np.exp(
                        entropy[
                            active
                        ]
                    )
                )
            ),

        "mean_mixture_probability":
            [
                float(
                    value
                )
                for value in (
                    np.mean(
                        probabilities[
                            active
                        ],
                        axis=0,
                    ).tolist()
                )
            ],

        "MAP_mode_counts":
            {
                str(
                    index
                ):
                    int(
                        map_counts[
                            index
                        ]
                    )
                for index in range(
                    3
                )
            },
    }

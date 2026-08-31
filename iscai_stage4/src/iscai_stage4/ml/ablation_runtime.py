from __future__ import annotations

import ast
import inspect
from pathlib import Path
import re

import numpy as np
import torch

from iscai_stage4.data import neural_inputs


FEATURE_DIM = int(
    neural_inputs.FEATURE_DIM
)

EXPECTED_FEATURE_DIM = 14

ABLATION_FULL = "full"
ABLATION_NO_R = "no_measurement_covariance"
ABLATION_ACTOR_ONLY = "actor_only"
ABLATION_NO_MAP = "no_map"


def _sequence_candidate(
    value,
):
    if isinstance(
        value,
        (
            tuple,
            list,
        ),
    ):
        if (
            len(value)
            ==
            FEATURE_DIM
            and
            all(
                isinstance(
                    item,
                    str,
                )
                for item in value
            )
        ):
            return tuple(
                value
            )

    if isinstance(
        value,
        np.ndarray,
    ):
        if (
            value.ndim == 1
            and
            len(value) == FEATURE_DIM
            and
            all(
                isinstance(
                    item,
                    str,
                )
                for item in value.tolist()
            )
        ):
            return tuple(
                value.tolist()
            )

    if isinstance(
        value,
        dict,
    ):
        if len(value) != FEATURE_DIM:
            return None

        # name -> index
        if (
            all(
                isinstance(
                    key,
                    str,
                )
                for key in value
            )
            and
            all(
                isinstance(
                    item,
                    int,
                )
                for item in value.values()
            )
            and
            set(
                value.values()
            )
            ==
            set(
                range(
                    FEATURE_DIM
                )
            )
        ):
            ordered = [
                None
            ] * FEATURE_DIM

            for name, index in (
                value.items()
            ):
                ordered[
                    index
                ] = name

            return tuple(
                ordered
            )

        # index -> name
        if (
            all(
                isinstance(
                    key,
                    int,
                )
                for key in value
            )
            and
            set(value)
            ==
            set(
                range(
                    FEATURE_DIM
                )
            )
            and
            all(
                isinstance(
                    item,
                    str,
                )
                for item in value.values()
            )
        ):
            return tuple(
                value[index]
                for index in range(
                    FEATURE_DIM
                )
            )

    return None


def _candidate_score(
    names,
):
    text = " ".join(
        name.lower()
        for name in names
    )

    score = 0

    for token in (
        "range",
        "az",
        "el",
        "radial",
        "velocity",
        "position",
        "observ",
        "valid",
        "cov",
        "variance",
        "sigma",
        "uncert",
    ):
        if token in text:
            score += 1

    return score


def discover_history_feature_names():
    """
    Discover the existing 14-feature causal
    history contract without hard-coding indices.

    Sources checked:
      1. live symbols in neural_inputs.py
      2. literal list/tuple/dict assignments in source

    If the mapping is ambiguous, fail explicitly
    rather than guessing.
    """

    if FEATURE_DIM != EXPECTED_FEATURE_DIM:
        raise RuntimeError(
            "Frozen Stage4 feature dimension "
            f"changed: {FEATURE_DIM}"
        )

    candidates = []

    for symbol, value in vars(
        neural_inputs
    ).items():
        candidate = (
            _sequence_candidate(
                value
            )
        )

        if candidate is not None:
            candidates.append(
                (
                    f"runtime:{symbol}",
                    candidate,
                )
            )

    source_path = Path(
        inspect.getsourcefile(
            neural_inputs
        )
    )

    source = source_path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source
    )

    for node in ast.walk(
        tree
    ):
        value_node = None
        target_name = None

        if isinstance(
            node,
            ast.Assign,
        ):
            if len(
                node.targets
            ) == 1:
                target = node.targets[0]

                if isinstance(
                    target,
                    ast.Name,
                ):
                    target_name = (
                        target.id
                    )

            value_node = node.value

        elif isinstance(
            node,
            ast.AnnAssign,
        ):
            if isinstance(
                node.target,
                ast.Name,
            ):
                target_name = (
                    node.target.id
                )

            value_node = node.value

        if (
            target_name is None
            or
            value_node is None
        ):
            continue

        try:
            literal = (
                ast.literal_eval(
                    value_node
                )
            )
        except Exception:
            continue

        candidate = (
            _sequence_candidate(
                literal
            )
        )

        if candidate is not None:
            candidates.append(
                (
                    f"source:{target_name}",
                    candidate,
                )
            )

    unique = {}

    for origin, names in (
        candidates
    ):
        unique.setdefault(
            names,
            [],
        ).append(
            origin
        )

    if not unique:
        raise RuntimeError(
            "No explicit 14-feature name "
            "mapping could be discovered "
            "from neural_inputs.py."
        )

    ranked = sorted(
        (
            (
                _candidate_score(
                    names
                ),
                names,
                origins,
            )
            for names, origins
            in unique.items()
        ),
        key=lambda item:
            item[0],
        reverse=True,
    )

    best_score = ranked[
        0
    ][
        0
    ]

    best = [
        item
        for item in ranked
        if item[0]
        ==
        best_score
    ]

    if len(best) != 1:
        alternatives = [
            {
                "score":
                    score,

                "names":
                    names,

                "origins":
                    origins,
            }
            for score, names, origins
            in best
        ]

        raise RuntimeError(
            "Ambiguous feature-name mapping: "
            +
            repr(
                alternatives
            )
        )

    (
        _,
        names,
        origins,
    ) = best[0]

    return {
        "names":
            tuple(
                names
            ),

        "origins":
            tuple(
                origins
            ),

        "source_path":
            str(
                source_path
            ),
    }


def is_measurement_covariance_feature(
    name,
):
    """
    Conservative semantic detector.

    We deliberately require uncertainty/
    covariance terminology rather than
    guessing from numeric feature position.
    """

    normalized = (
        str(name)
        .strip()
        .lower()
    )

    tokens = tuple(
        token
        for token in re.split(
            r"[^a-z0-9]+",
            normalized,
        )
        if token
    )

    explicit_tokens = {
        "cov",
        "covariance",
        "variance",
        "var",
        "sigma",
        "uncertainty",
        "uncert",
    }

    if any(
        token in explicit_tokens
        for token in tokens
    ):
        return True

    if (
        normalized.startswith(
            "measurement_cov"
        )
        or
        normalized.startswith(
            "measurement_var"
        )
        or
        normalized.startswith(
            "measurement_sigma"
        )
        or
        normalized.startswith(
            "log_cov"
        )
        or
        normalized.startswith(
            "log_var"
        )
        or
        normalized.startswith(
            "log_sigma"
        )
    ):
        return True

    # Matrix-style names such as R_00, R11.
    compact = re.sub(
        r"[^a-z0-9]",
        "",
        normalized,
    )

    if re.fullmatch(
        r"(log)?r(00|11|22|33)",
        compact,
    ):
        return True

    return False


def measurement_covariance_indices(
    feature_names,
):
    if len(
        feature_names
    ) != FEATURE_DIM:
        raise ValueError(
            "Feature-name count does "
            "not match FEATURE_DIM."
        )

    indices = tuple(
        index
        for index, name
        in enumerate(
            feature_names
        )
        if (
            is_measurement_covariance_feature(
                name
            )
        )
    )

    if not indices:
        raise RuntimeError(
            "No measurement-covariance "
            "feature could be identified "
            "semantically."
        )

    return indices


def apply_normalized_ablation(
    target_history,
    neighbor_histories,
    neighbor_mask,
    map_context,
    *,
    mode,
    covariance_indices,
):
    """
    Ablations are applied AFTER frozen fit-only
    normalization.

    Therefore setting a continuous feature to zero
    is fit-mean imputation, NOT raw numerical zero.

    This avoids creating out-of-distribution raw
    covariance values merely to remove R_t.
    """

    valid_modes = {
        ABLATION_FULL,
        ABLATION_NO_R,
        ABLATION_ACTOR_ONLY,
        ABLATION_NO_MAP,
    }

    if mode not in valid_modes:
        raise ValueError(
            f"Unknown ablation mode: {mode}"
        )

    if (
        target_history.shape[-1]
        !=
        FEATURE_DIM
    ):
        raise ValueError(
            "Target feature dimension changed."
        )

    if (
        neighbor_histories.shape[-1]
        !=
        FEATURE_DIM
    ):
        raise ValueError(
            "Neighbour feature dimension changed."
        )

    target = target_history
    neighbors = neighbor_histories
    mask = neighbor_mask
    map_value = map_context

    if mode == ABLATION_NO_R:
        target = (
            target_history.clone()
        )

        neighbors = (
            neighbor_histories.clone()
        )

        target[
            ...,
            list(
                covariance_indices
            )
        ] = 0.0

        neighbors[
            ...,
            list(
                covariance_indices
            )
        ] = 0.0

    elif mode == ABLATION_ACTOR_ONLY:
        neighbors = torch.zeros_like(
            neighbor_histories
        )

        mask = torch.zeros_like(
            neighbor_mask
        )

    elif mode == ABLATION_NO_MAP:
        map_value = torch.zeros_like(
            map_context
        )

    return (
        target,
        neighbors,
        mask,
        map_value,
    )

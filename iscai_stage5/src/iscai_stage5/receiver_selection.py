from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


PRIMARY_POLICY = (
    "nearest_causal_vehicle_ahead"
)

CONNECTIVITY_SEMANTICS = (
    "constructed_hypothetical_connected_vehicle"
)

NO_ELIGIBLE_RECEIVER = (
    "NO_ELIGIBLE_RECEIVER"
)

SELECTED_RECEIVER = (
    "SELECTED"
)


VEHICLE_CLASS_ALIASES = frozenset(
    {
        "vehicle",
        "type_vehicle",
        "car",
    }
)


@dataclass(
    frozen=True
)
class ReceiverCandidate:
    """
    Causal current-time candidate for the communication
    receiver.

    No future state, tracks_to_predict flag, oracle
    connectivity label or perfect track ID enters this
    contract.

    `position_h0_m` is the current associated actor-centre
    position in the frozen headlamp frame H0.
    """

    semantic_class: str

    position_h0_m: tuple[
        float,
        float,
        float,
    ]

    current_available: bool = True

    association_valid: bool = True


@dataclass(
    frozen=True
)
class ReceiverSelectionConfig:
    """
    Primary causal receiver eligibility settings.

    `max_planar_range_m=None` deliberately means that the
    Stage5 development-only receiver-range cutoff has NOT
    yet been frozen.  That numerical value must be selected
    only from non-formal development evidence before the
    formal N=120 evaluation.
    """

    minimum_forward_h0_m: float = 0.0

    max_planar_range_m: float | None = None


@dataclass(
    frozen=True
)
class ReceiverSelectionResult:
    status: str

    selected_candidate_index: int | None

    planar_range_m: float | None

    forward_h0_m: float | None

    lateral_h0_m: float | None

    eligible_candidate_count: int

    primary_policy: str = PRIMARY_POLICY

    connectivity_semantics: str = (
        CONNECTIVITY_SEMANTICS
    )


def normalize_semantic_class(
    value: str,
) -> str:
    return (
        str(
            value
        )
        .strip()
        .lower()
        .replace(
            " ",
            "_",
        )
        .replace(
            "-",
            "_",
        )
    )


def is_vehicle_semantic_class(
    value: str,
) -> bool:
    return (
        normalize_semantic_class(
            value
        )
        in
        VEHICLE_CLASS_ALIASES
    )


def _validate_position(
    position_h0_m: tuple[
        float,
        float,
        float,
    ],
) -> tuple[
    float,
    float,
    float,
]:
    if len(
        position_h0_m
    ) != 3:
        raise ValueError(
            "Receiver candidate position must "
            "contain exactly x/y/z."
        )

    values = tuple(
        float(
            value
        )
        for value in position_h0_m
    )

    if not all(
        math.isfinite(
            value
        )
        for value in values
    ):
        raise ValueError(
            "Receiver candidate position "
            "must be finite."
        )

    return values


def _validate_config(
    config: ReceiverSelectionConfig,
) -> None:
    if not math.isfinite(
        float(
            config.minimum_forward_h0_m
        )
    ):
        raise ValueError(
            "minimum_forward_h0_m must be finite."
        )

    if (
        config.max_planar_range_m
        is not None
    ):
        maximum = float(
            config.max_planar_range_m
        )

        if (
            not math.isfinite(
                maximum
            )
            or
            maximum <= 0.0
        ):
            raise ValueError(
                "max_planar_range_m must be "
                "positive and finite when set."
            )


def _eligible(
    candidate: ReceiverCandidate,
    config: ReceiverSelectionConfig,
) -> tuple[
    bool,
    float,
    float,
    float,
]:
    x, y, _ = _validate_position(
        candidate.position_h0_m
    )

    planar_range = math.hypot(
        x,
        y,
    )

    if not candidate.current_available:
        return (
            False,
            planar_range,
            x,
            y,
        )

    if not candidate.association_valid:
        return (
            False,
            planar_range,
            x,
            y,
        )

    if not is_vehicle_semantic_class(
        candidate.semantic_class
    ):
        return (
            False,
            planar_range,
            x,
            y,
        )

    #
    # Strictly forward in H0.
    #
    # x == minimum_forward_h0_m is not considered
    # "ahead".
    #
    if not (
        x
        >
        float(
            config.minimum_forward_h0_m
        )
    ):
        return (
            False,
            planar_range,
            x,
            y,
        )

    if (
        config.max_planar_range_m
        is not None
        and
        planar_range
        >
        float(
            config.max_planar_range_m
        )
    ):
        return (
            False,
            planar_range,
            x,
            y,
        )

    return (
        True,
        planar_range,
        x,
        y,
    )


def select_primary_receiver(
    candidates: Sequence[
        ReceiverCandidate
    ],
    config: ReceiverSelectionConfig | None = None,
) -> ReceiverSelectionResult:
    """
    Select the primary Stage5 communication receiver.

    Primary policy:
        nearest causal vehicle ahead in H0.

    Ranking uses only current causal geometry and actor
    semantic class.

    The candidate list index is used only as the final
    deterministic tie-break when causal geometry is exactly
    tied.  It is not a physical/model feature and it is not
    a WOMD perfect track ID.
    """

    if config is None:
        config = ReceiverSelectionConfig()

    _validate_config(
        config
    )

    eligible = []

    for index, candidate in enumerate(
        candidates
    ):
        (
            valid,
            planar_range,
            x,
            y,
        ) = _eligible(
            candidate,
            config,
        )

        if not valid:
            continue

        #
        # Geometric deterministic ordering:
        #
        # 1. nearest planar range,
        # 2. smallest |lateral displacement|,
        # 3. furthest forward component if otherwise tied,
        # 4. signed lateral coordinate,
        # 5. current candidate list index only as final
        #    exact-tie resolution.
        #
        score = (
            planar_range,
            abs(
                y
            ),
            -x,
            y,
            index,
        )

        eligible.append(
            (
                score,
                index,
                planar_range,
                x,
                y,
            )
        )

    if not eligible:
        return ReceiverSelectionResult(
            status=(
                NO_ELIGIBLE_RECEIVER
            ),

            selected_candidate_index=None,

            planar_range_m=None,

            forward_h0_m=None,

            lateral_h0_m=None,

            eligible_candidate_count=0,
        )

    eligible.sort(
        key=lambda item:
            item[
                0
            ]
    )

    (
        _,
        selected_index,
        selected_range,
        selected_x,
        selected_y,
    ) = eligible[
        0
    ]

    return ReceiverSelectionResult(
        status=(
            SELECTED_RECEIVER
        ),

        selected_candidate_index=(
            int(
                selected_index
            )
        ),

        planar_range_m=(
            float(
                selected_range
            )
        ),

        forward_h0_m=(
            float(
                selected_x
            )
        ),

        lateral_h0_m=(
            float(
                selected_y
            )
        ),

        eligible_candidate_count=(
            len(
                eligible
            )
        ),
    )

from __future__ import annotations

from dataclasses import replace
from functools import wraps
from time import perf_counter
import numpy as np


DEFAULT_BUDGET_FRACTION = 0.02
_DIM_EPS = 1.0
_VRU_FLOOR_EPS = 1.0e-12

_LATENCY_S: list[float] = []
_SELECTED_COUNTS: list[int] = []


def reset_runtime_statistics() -> None:
    _LATENCY_S.clear()
    _SELECTED_COUNTS.clear()


def runtime_statistics() -> dict:
    x = np.asarray(_LATENCY_S, dtype=float)

    if x.size == 0:
        latency = {
            "calls": 0,
            "mean_s": None,
            "median_s": None,
            "p95_s": None,
            "max_s": None,
        }
    else:
        latency = {
            "calls": int(x.size),
            "mean_s": float(np.mean(x)),
            "median_s": float(np.median(x)),
            "p95_s": float(np.quantile(x, 0.95)),
            "max_s": float(np.max(x)),
        }

    return {
        "projection_latency": latency,
        "selected_extra_cell_count": {
            "calls": len(_SELECTED_COUNTS),
            "mean": (
                float(np.mean(_SELECTED_COUNTS))
                if _SELECTED_COUNTS
                else None
            ),
            "max": (
                int(max(_SELECTED_COUNTS))
                if _SELECTED_COUNTS
                else None
            ),
        },
    }


def _current_map(current_illumination, old_schedule):
    current = np.asarray(
        current_illumination,
        dtype=float,
    )

    old = np.asarray(
        old_schedule,
        dtype=float,
    )

    if old.ndim < 2:
        raise ValueError(
            "Rate-limited predictive schedule must have "
            "a horizon dimension plus actuator dimensions."
        )

    if current.shape == old.shape[1:]:
        base = current

    elif current.shape == old.shape:
        if not np.allclose(
            current,
            current[0][None, ...],
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                "Current illumination supplied as a schedule "
                "but is not a held causal t0 map."
            )

        base = current[0]

    else:
        raise ValueError(
            "current_illumination shape does not match "
            "RateLimitedSchedule.illumination."
        )

    if not np.all(np.isfinite(base)):
        raise ValueError(
            "Current illumination contains non-finite values."
        )

    if not np.all(np.isfinite(old)):
        raise ValueError(
            "Predictive illumination contains non-finite values."
        )

    if np.any(base < 0.0) or np.any(base > 1.0):
        raise ValueError(
            "Current illumination must lie in [0,1]."
        )

    if np.any(old < 0.0) or np.any(old > 1.0):
        raise ValueError(
            "Predictive illumination must lie in [0,1]."
        )

    return base, old


def _vru_protected_spatial_mask(
    vru_floor_guard,
    *,
    horizon_count: int,
    actuator_shape: tuple[int, ...],
) -> np.ndarray:
    if vru_floor_guard is None:
        return np.zeros(
            actuator_shape,
            dtype=bool,
        )

    guard = np.asarray(
        vru_floor_guard,
        dtype=float,
    )

    if guard.shape == actuator_shape:
        guard = np.broadcast_to(
            guard,
            (horizon_count,) + actuator_shape,
        )

    elif guard.shape != (
        (horizon_count,) + actuator_shape
    ):
        raise ValueError(
            "vru_floor_guard shape is incompatible with "
            "the predictive illumination schedule."
        )

    if not np.all(np.isfinite(guard)):
        raise ValueError(
            "vru_floor_guard contains non-finite values."
        )

    # Stage6 V1 uses floor=1.0 for pedestrian/cyclist
    # visibility protection.  Exclude every spatial cell that
    # is fully protected at ANY prediction horizon.
    return np.any(
        guard >= (1.0 - _VRU_FLOOR_EPS),
        axis=0,
    )


def budgeted_residual_projection(
    rate_limited_result,
    *,
    current_illumination,
    vru_floor_guard=None,
    budget_fraction: float = DEFAULT_BUDGET_FRACTION,
):
    """
    Stage6-V2 predictive residual safety controller.

    Baseline:
      exact held causal original-reactive Part-A t0 map.

    Predictive additions:
      only spatial cells that
        1) are not already dimmed by the reactive map,
        2) are dimmed by the frozen V1 predictive schedule,
        3) are not protected by the VRU floor guard.

    The selected spatial set is STATIC across all horizons.
    Each selected cell keeps its complete V1 post-smoothing,
    post-rate-limit time sequence. Every other cell remains
    exactly at the causal reactive t0 intensity.

    No oracle, future GT, evaluator mask, track truth or
    acceptance result enters this function.
    """

    q = float(budget_fraction)

    if not np.isfinite(q):
        raise ValueError(
            "budget_fraction must be finite."
        )

    if q < 0.0 or q > 1.0:
        raise ValueError(
            "budget_fraction must lie in [0,1]."
        )

    old = np.asarray(
        rate_limited_result.illumination,
        dtype=float,
    )

    base, old = _current_map(
        current_illumination,
        old,
    )

    H = int(old.shape[0])
    actuator_shape = tuple(
        int(x)
        for x in old.shape[1:]
    )

    cell_count = int(
        np.prod(
            actuator_shape,
            dtype=np.int64,
        )
    )

    budget_cells = int(
        np.floor(
            q * cell_count
        )
    )

    reactive_dim = (
        base < _DIM_EPS
    )

    predictive_dim = (
        old < _DIM_EPS
    )

    vru_protected = (
        _vru_protected_spatial_mask(
            vru_floor_guard,
            horizon_count=H,
            actuator_shape=actuator_shape,
        )
    )

    predictive_extra = (
        (~reactive_dim)
        &
        np.any(
            predictive_dim,
            axis=0,
        )
        &
        (~vru_protected)
    )

    flat_candidate = np.flatnonzero(
        predictive_extra.reshape(-1)
    )

    depth = np.max(
        1.0 - old,
        axis=0,
    ).reshape(-1)

    persistence = np.sum(
        predictive_dim,
        axis=0,
        dtype=np.int64,
    ).reshape(-1)

    if flat_candidate.size:
        # Deterministic total order:
        #   1. deeper predictive dimming,
        #   2. more persistent across horizons,
        #   3. lower canonical flat actuator index.
        order = np.lexsort(
            (
                flat_candidate,
                -persistence[
                    flat_candidate
                ],
                -depth[
                    flat_candidate
                ],
            )
        )

        selected_flat = flat_candidate[
            order[
                :min(
                    budget_cells,
                    flat_candidate.size,
                )
            ]
        ]

    else:
        selected_flat = np.empty(
            0,
            dtype=np.int64,
        )

    selected = np.zeros(
        cell_count,
        dtype=bool,
    )

    selected[
        selected_flat
    ] = True

    selected = selected.reshape(
        actuator_shape
    )

    new = np.broadcast_to(
        base,
        old.shape,
    ).copy()

    old_2d = old.reshape(
        H,
        cell_count,
    )

    new_2d = new.reshape(
        H,
        cell_count,
    )

    if selected_flat.size:
        new_2d[
            :,
            selected_flat,
        ] = old_2d[
            :,
            selected_flat,
        ]

    # ---------------------------------------------------------
    # Distribution-free hard over-mask budget proof.
    # ---------------------------------------------------------

    extra_dim = (
        (new < 1.0)
        &
        (~reactive_dim)[
            None,
            ...
        ]
    )

    extra_per_horizon = np.count_nonzero(
        extra_dim.reshape(
            H,
            cell_count,
        ),
        axis=1,
    )

    if np.any(
        extra_per_horizon
        > budget_cells
    ):
        raise RuntimeError(
            "Stage6-V2 residual budget invariant violated."
        )

    if selected_flat.size > budget_cells:
        raise RuntimeError(
            "Stage6-V2 selected spatial support exceeds budget."
        )

    if np.any(
        selected
        &
        vru_protected
    ):
        raise RuntimeError(
            "Stage6-V2 selected a VRU-protected cell."
        )

    # ---------------------------------------------------------
    # The temporal/rate-validity argument is structural:
    #
    # each cell is either
    #   A) the exact held causal current value, or
    #   B) the exact old V1 post-smoothing/post-rate-limit
    #      sequence for that cell.
    #
    # No new temporal trajectory is synthesized here.
    # ---------------------------------------------------------

    old_override = np.asarray(
        rate_limited_result.safety_override_mask,
        dtype=bool,
    )

    if old_override.shape != old.shape:
        raise ValueError(
            "safety_override_mask shape mismatch."
        )

    new_override = (
        old_override
        &
        selected[
            None,
            ...
        ]
    )

    new_override_count = int(
        np.count_nonzero(
            new_override
        )
    )

    new_override_fraction = float(
        new_override_count
        /
        new_override.size
    )

    _SELECTED_COUNTS.append(
        int(
            selected_flat.size
        )
    )

    semantics = (
        str(
            rate_limited_result.semantics
        )
        + "; Stage6-V2 causal budgeted predictive residual; "
        + f"q_budget={q:.17g}; "
        + f"max_extra_spatial_cells={budget_cells}; "
        + "VRU-floor-guard exclusion; "
        + "no evaluator/oracle input"
    )

    return replace(
        rate_limited_result,
        illumination=new,
        safety_override_mask=new_override,
        safety_override_count=
            new_override_count,
        safety_override_fraction=
            new_override_fraction,
        semantics=semantics,
    )


def make_budgeted_apply(
    original_apply_actuation_rate_limit,
    *,
    budget_fraction: float = DEFAULT_BUDGET_FRACTION,
):
    @wraps(
        original_apply_actuation_rate_limit
    )
    def wrapped(*args, **kwargs):
        if (
            "current_illumination"
            not in kwargs
        ):
            raise RuntimeError(
                "Stage6-V2 requires the explicit causal "
                "current_illumination argument."
            )

        result = (
            original_apply_actuation_rate_limit(
                *args,
                **kwargs,
            )
        )

        t0 = perf_counter()

        out = budgeted_residual_projection(
            result,
            current_illumination=
                kwargs[
                    "current_illumination"
                ],
            vru_floor_guard=
                kwargs.get(
                    "vru_floor_guard"
                ),
            budget_fraction=
                budget_fraction,
        )

        _LATENCY_S.append(
            float(
                perf_counter()
                - t0
            )
        )

        return out

    return wrapped

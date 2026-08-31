
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


class ScientificBoundaryError(
    RuntimeError
):
    pass


@dataclass(frozen=True)
class PredictionEvent:

    scenario_id: str
    prediction_id: str
    horizon_s: float

    predicted_xy_m: tuple[float, float]


class PredictionBeforeTruthLedger:
    """
    Repaired Stage-4 scientific boundary.

    Predictions are algorithm-side.

    Evaluator truth becomes legal only
    after seal_predictions().
    """

    def __init__(self):

        self._events = []
        self._sealed = False
        self._truth_allowed = False


    @property
    def sealed(self):

        return self._sealed


    @property
    def truth_allowed(self):

        return self._truth_allowed


    @property
    def events(self):

        return tuple(
            self._events
        )


    def add_prediction(
        self,
        event,
    ):

        if self._sealed:
            raise ScientificBoundaryError(
                "Prediction added after "
                "evaluator boundary."
            )

        if self._truth_allowed:
            raise ScientificBoundaryError(
                "Prediction added after "
                "truth became available."
            )

        self._events.append(
            event
        )


    def seal_predictions(self):

        self._sealed = True


    def allow_evaluator_truth(self):

        if not self._sealed:
            raise ScientificBoundaryError(
                "Evaluator truth requested "
                "before prediction seal."
            )

        self._truth_allowed = True


def _key(
    row,
    fields,
):

    values = []

    for field in fields:

        if field not in row:
            raise KeyError(
                "Missing common-support "
                f"field: {field}"
            )

        value = row[field]

        if field == "horizon_s":
            value = round(
                float(value),
                12,
            )

        values.append(
            value
        )

    return tuple(
        values
    )


def _index(
    rows,
    fields,
):

    result = {}

    for row in rows:

        key = _key(
            row,
            fields,
        )

        if key in result:
            raise ScientificBoundaryError(
                "Duplicate common-support "
                f"key: {key!r}"
            )

        result[key] = row

    return result


def _xy(value):

    value = np.asarray(
        value,
        dtype=np.float64,
    )

    if value.shape != (2,):
        raise ValueError(
            "XY must have shape (2,)."
        )

    if not np.all(
        np.isfinite(value)
    ):
        raise ValueError(
            "Non-finite XY."
        )

    return value


def paired_common_support_metrics(
    neural_rows: Iterable[Mapping],
    classical_rows: Iterable[Mapping],
    *,
    key_fields: Sequence[str] = (
            "scenario_id",
            "target_id",
            "horizon_s",
        ),
):

    """
    Apples-to-apples comparison.

    No scalar ADE comparison is allowed
    across different matched populations.

    Only the exact common
    scenario/target/horizon intersection
    contributes.
    """

    neural = _index(
        neural_rows,
        key_fields,
    )

    classical = _index(
        classical_rows,
        key_fields,
    )

    common = sorted(
        set(neural).intersection(
            classical
        ),
        key=repr,
    )

    if not common:
        raise ScientificBoundaryError(
            "No common support."
        )

    neural_error = []
    classical_error = []

    per_horizon = {}

    for key in common:

        nr = neural[key]
        cr = classical[key]

        nt = _xy(
            nr["truth_xy_m"]
        )

        ct = _xy(
            cr["truth_xy_m"]
        )

        if not np.array_equal(
            nt,
            ct,
        ):
            raise ScientificBoundaryError(
                "Truth mismatch on "
                f"{key!r}"
            )

        npred = _xy(
            nr["predicted_xy_m"]
        )

        cpred = _xy(
            cr["predicted_xy_m"]
        )

        ne = float(
            np.linalg.norm(
                npred - nt
            )
        )

        ce = float(
            np.linalg.norm(
                cpred - nt
            )
        )

        neural_error.append(
            ne
        )

        classical_error.append(
            ce
        )

        horizon = round(
            float(
                nr["horizon_s"]
            ),
            12,
        )

        bucket = (
            per_horizon.setdefault(
                horizon,
                {
                    "neural": [],
                    "classical": [],
                },
            )
        )

        bucket["neural"].append(
            ne
        )

        bucket["classical"].append(
            ce
        )

    n_ade = float(
        np.mean(
            neural_error
        )
    )

    c_ade = float(
        np.mean(
            classical_error
        )
    )

    horizons = {}

    for horizon, values in sorted(
        per_horizon.items()
    ):

        horizons[str(horizon)] = {
            "n":
                len(
                    values["neural"]
                ),
            "neural_mean_error_m":
                float(
                    np.mean(
                        values["neural"]
                    )
                ),
            "classical_mean_error_m":
                float(
                    np.mean(
                        values[
                            "classical"
                        ]
                    )
                ),
        }

    return {
        "support":
            "exact_common_"
            "scenario_target_horizon",
        "common_event_count":
            len(common),
        "neural_ADE_m":
            n_ade,
        "classical_ADE_m":
            c_ade,
        "neural_minus_classical_ADE_m":
            n_ade - c_ade,
        "neural_beats_classical":
            n_ade < c_ade,
        "per_horizon":
            horizons,
    }


def reject_cross_population_scalar_claim(
    *,
    neural_support,
    classical_support,
    common_support=None,
):

    if (
        int(neural_support)
        != int(classical_support)
    ):

        if (
            common_support is None
            or int(common_support) <= 0
        ):

            raise ScientificBoundaryError(
                "Different matched populations "
                "require explicit "
                "common support."
            )


def assert_causal_formal_source(
    source,
):

    """
    Repaired formal inference source must
    not use attach_supervision before
    predictions.

    Checker will apply this to the new
    repaired inference route.
    """

    if "attach_supervision(" in source:

        raise ScientificBoundaryError(
            "Repaired inference source "
            "contains attach_supervision()."
        )

    prediction_marker = (
        "PREDICTIONS_SEALED_BEFORE_"
        "EVALUATOR_TRUTH"
    )

    truth_marker = (
        "EVALUATOR_TRUTH_READ_"
        "AFTER_PREDICTIONS"
    )

    p = source.find(
        prediction_marker
    )

    t = source.find(
        truth_marker
    )

    if (
        p < 0
        or t < 0
        or p >= t
    ):
        raise ScientificBoundaryError(
            "Prediction/truth phase "
            "markers missing or reversed."
        )

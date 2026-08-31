#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/agni/waymo")
S3 = ROOT / "iscai_stage3"
S4 = ROOT / "iscai_stage4"

AUDIT = ROOT / "audits" / "stage4_repair_v1"
BACKUP = AUDIT / "historical_pre_repair_science"
REPORT = AUDIT / "repair_report.json"

EXPECTED_STAGE3_RUN_SHA = (
    "9151ef8714fb710b57b8cc762bb809ce"
    "bf29b6174d1aeb8dab609b42f6641e7b"
)

EXPECTED_STAGE3_MANIFEST_SHA = (
    "2208e7287ddf6439fda4597c435a9cba"
    "1d1b9d0e4c4547bc5dd92e56e8124e46"
)

EXPECTED_OLD_STAGE4_CLOSURE_SHA = (
    "570da4feb918b1025b5e85cc919360d9"
    "22b471c468c85b3844f13fb7774e7c2f"
)

EXPECTED_OLD_STAGE4_FORMAL_SHA = (
    "402ab3d771e9a3c77de0c091c893e9a"
    "0d6ee5826ec52d07d27b919c68276b353"
)

MODULE = (
    S4
    / "src"
    / "iscai_stage4"
    / "ml"
    / "scientific_repair_v1.py"
)

TEST = (
    S4
    / "tests"
    / "test_stage4_repair_v1_semantics.py"
)

CONFIG = (
    S4
    / "configs"
    / "stage4_scientific_repair_v1.json"
)

STATUS = (
    S4
    / "reports"
    / "stage4_repair_v1_status.json"
)


class RepairError(RuntimeError):
    pass


def ensure(value, message):
    if not value:
        raise RepairError(message)


def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def read_json(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_name(
        path.name + ".stage4_repair_tmp"
    )

    temp.write_text(
        text,
        encoding="utf-8",
    )

    os.replace(
        temp,
        path,
    )


def write_json(path, value):
    write_text(
        path,
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def preserve(rel):
    src = S4 / rel

    if not src.is_file():
        return {
            "path": str(src),
            "missing": True,
        }

    dst = BACKUP / rel

    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not dst.exists():
        shutil.copy2(
            src,
            dst,
        )
    else:
        ensure(
            sha256(src) == sha256(dst),
            "Historical backup differs: "
            + str(dst),
        )

    return {
        "source": str(src),
        "backup": str(dst),
        "sha256": sha256(src),
    }


def verify_repaired_stage3():

    closure = (
        S3
        / "reports"
        / "stage3_final_closure.json"
    )

    formal = (
        S3
        / "reports"
        / "block38f_formal_evaluation.json"
    )

    ensure(
        closure.is_file(),
        "Missing repaired Stage3 closure.",
    )

    ensure(
        formal.is_file(),
        "Missing repaired Stage3 formal report.",
    )

    closure_json = read_json(
        closure
    )

    ensure(
        closure_json.get("status")
        == "COMPLETE_FROZEN",
        "Stage3 is not COMPLETE_FROZEN.",
    )

    formal_text = formal.read_text(
        encoding="utf-8"
    )

    ensure(
        EXPECTED_STAGE3_RUN_SHA
        in formal_text,
        "Repaired Stage3 run SHA mismatch.",
    )

    ensure(
        EXPECTED_STAGE3_MANIFEST_SHA
        in formal_text,
        "Stage3 formal manifest SHA mismatch.",
    )

    return {
        "closure_sha256":
            sha256(closure),
        "formal_sha256":
            sha256(formal),
        "run_sha256":
            EXPECTED_STAGE3_RUN_SHA,
        "manifest_sha256":
            EXPECTED_STAGE3_MANIFEST_SHA,
    }


MODULE_SOURCE = r'''
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
'''


TEST_SOURCE = r'''
from __future__ import annotations

import unittest

from iscai_stage4.ml.scientific_repair_v1 import (
    PredictionBeforeTruthLedger,
    PredictionEvent,
    ScientificBoundaryError,
    assert_causal_formal_source,
    paired_common_support_metrics,
    reject_cross_population_scalar_claim,
)


class TestStage4ScientificRepairV1(
    unittest.TestCase
):

    def test_truth_before_prediction_seal_fails(
        self
    ):

        ledger = (
            PredictionBeforeTruthLedger()
        )

        with self.assertRaises(
            ScientificBoundaryError
        ):
            ledger.allow_evaluator_truth()


    def test_prediction_after_truth_fails(
        self
    ):

        ledger = (
            PredictionBeforeTruthLedger()
        )

        ledger.add_prediction(
            PredictionEvent(
                scenario_id="s",
                prediction_id="p",
                horizon_s=0.1,
                predicted_xy_m=(
                    1.0,
                    2.0,
                ),
            )
        )

        ledger.seal_predictions()
        ledger.allow_evaluator_truth()

        with self.assertRaises(
            ScientificBoundaryError
        ):
            ledger.add_prediction(
                PredictionEvent(
                    scenario_id="s",
                    prediction_id="q",
                    horizon_s=0.1,
                    predicted_xy_m=(
                        0.0,
                        0.0,
                    ),
                )
            )


    def test_common_support_only(
        self
    ):

        neural = [
            {
                "scenario_id": "s",
                "target_id": "a",
                "horizon_s": 0.1,
                "predicted_xy_m":
                    [1.0, 0.0],
                "truth_xy_m":
                    [0.0, 0.0],
            },
            {
                "scenario_id": "s",
                "target_id": "b",
                "horizon_s": 0.1,
                "predicted_xy_m":
                    [100.0, 0.0],
                "truth_xy_m":
                    [0.0, 0.0],
            },
        ]

        classical = [
            {
                "scenario_id": "s",
                "target_id": "a",
                "horizon_s": 0.1,
                "predicted_xy_m":
                    [2.0, 0.0],
                "truth_xy_m":
                    [0.0, 0.0],
            }
        ]

        result = (
            paired_common_support_metrics(
                neural,
                classical,
            )
        )

        self.assertEqual(
            result[
                "common_event_count"
            ],
            1,
        )

        self.assertEqual(
            result[
                "neural_ADE_m"
            ],
            1.0,
        )

        self.assertEqual(
            result[
                "classical_ADE_m"
            ],
            2.0,
        )


    def test_cross_population_scalar_claim_fails(
        self
    ):

        with self.assertRaises(
            ScientificBoundaryError
        ):
            reject_cross_population_scalar_claim(
                neural_support=494,
                classical_support=465,
            )


    def test_common_support_allows_comparison(
        self
    ):

        reject_cross_population_scalar_claim(
            neural_support=494,
            classical_support=465,
            common_support=400,
        )


    def test_attach_supervision_rejected_from_repaired_route(
        self
    ):

        with self.assertRaises(
            ScientificBoundaryError
        ):
            assert_causal_formal_source(
                """
                PREDICTIONS_SEALED_BEFORE_EVALUATOR_TRUTH
                attach_supervision(...)
                EVALUATOR_TRUTH_READ_AFTER_PREDICTIONS
                """
            )


    def test_phase_order_passes(
        self
    ):

        assert_causal_formal_source(
            """
            PREDICTIONS_SEALED_BEFORE_EVALUATOR_TRUTH
            EVALUATOR_TRUTH_READ_AFTER_PREDICTIONS
            """
        )


if __name__ == "__main__":
    unittest.main()
'''


def main():

    print(
        "=" * 60
    )

    print(
        "STAGE 4 SCIENTIFIC REPAIR V1"
    )

    print(
        "=" * 60
    )

    ensure(
        S4.is_dir(),
        "Stage4 root missing.",
    )

    AUDIT.mkdir(
        parents=True,
        exist_ok=True,
    )

    BACKUP.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "A. Verify repaired Stage3"
    )

    stage3 = (
        verify_repaired_stage3()
    )

    print(
        "Stage3 = COMPLETE_FROZEN"
    )

    print(
        "Stage3 repaired run SHA =",
        stage3["run_sha256"],
    )

    old_closure = (
        S4
        / "reports"
        / "stage4_final_closure.json"
    )

    old_formal = (
        S4
        / "reports"
        / "block48_formal_evaluation.json"
    )

    ensure(
        old_closure.is_file(),
        "Old Stage4 closure missing.",
    )

    ensure(
        old_formal.is_file(),
        "Old Stage4 formal report missing.",
    )

    closure_sha = sha256(
        old_closure
    )

    formal_sha = sha256(
        old_formal
    )

    already_applied = all(
        p.is_file()
        for p in (
            MODULE,
            TEST,
            CONFIG,
            STATUS,
        )
    )

    print()
    print(
        "B. Verify audited Stage4 state"
    )

    if not already_applied:

        ensure(
            closure_sha
            == EXPECTED_OLD_STAGE4_CLOSURE_SHA,
            "Old Stage4 closure SHA "
            "changed since audit.",
        )

        ensure(
            formal_sha
            == EXPECTED_OLD_STAGE4_FORMAL_SHA,
            "Old Stage4 formal SHA "
            "changed since audit.",
        )

    print(
        "old closure SHA =",
        closure_sha,
    )

    print(
        "old formal SHA  =",
        formal_sha,
    )

    print()
    print(
        "C. Preserve historical evidence"
    )

    historical_files = [
        "reports/block48_formal_evaluation.json",
        "reports/block49_reproducibility.json",
        "reports/stage4_final_closure.json",
        "artifacts/block410/"
        "stage4_final_freeze_manifest.json",
        "artifacts/block410/"
        "stage4_to_stage5_handoff.json",
        "artifacts/block48/"
        "formal_cache_manifest.json",
    ]

    historical = {
        rel: preserve(rel)
        for rel in historical_files
    }

    print(
        "historical files =",
        len(historical),
    )

    print()
    print(
        "D. Install repaired scientific rules"
    )

    write_text(
        MODULE,
        MODULE_SOURCE,
    )

    write_text(
        TEST,
        TEST_SOURCE,
    )

    protocol = {
        "stage": 4,
        "repair":
            "stage4_scientific_repair_v1",
        "status":
            "REPAIR_APPLIED_PENDING_"
            "INDEPENDENT_CHECK",

        "training": False,
        "recalibration": False,
        "checkpoint_weights_modified":
            False,
        "normalization_modified":
            False,
        "split_modified":
            False,

        "mandatory_repairs": {

            "oracle_current_truth_prefilter": {
                "old":
                    "current WOMD truth "
                    "could filter estimated "
                    "tracks before prediction",

                "new":
                    "all causal predictions "
                    "must be generated and "
                    "sealed before evaluator "
                    "truth is read",

                "checker_must_verify":
                    True,
            },

            "classical_comparison": {
                "old":
                    "scalar ADE comparison "
                    "across different matched "
                    "populations",

                "new":
                    "exact common scenario-"
                    "target-horizon support",

                "required_stage3_run_sha256":
                    EXPECTED_STAGE3_RUN_SHA,
            },
        },

        "completion_gate": {
            "probabilistic_beats_at_least_"
            "one_classical_on_common_support":
                True,

            "confidence_regions_have_"
            "measured_calibration":
                True,
        },

        "calibration": {
            "reuse_existing_calibrator":
                True,

            "refit_on_formal":
                False,

            "measurement_R_t_modified":
                False,
        },

        "deferred_final_paper_scope": {
            "direct_noisy_PC_FMCW_vr_feature":
                "DEFERRED_REQUIRES_RETRAINING",

            "full_input_ablation_suite":
                "DEFERRED_TO_FINAL_"
                "EXPERIMENTAL_SYNTHESIS",
        },

        "upstream": {
            "Stage3_manifest_sha256":
                EXPECTED_STAGE3_MANIFEST_SHA,

            "Stage3_repaired_run_sha256":
                EXPECTED_STAGE3_RUN_SHA,
        },
    }

    write_json(
        CONFIG,
        protocol,
    )

    status = {
        "stage": 4,

        "repair":
            "stage4_scientific_repair_v1",

        "status":
            "REPAIR_APPLIED_PENDING_"
            "INDEPENDENT_CHECK",

        "legacy_Block48": {
            "sha256":
                formal_sha,
            "role":
                "HISTORICAL_PRE_REPAIR_ONLY",
        },

        "legacy_closure": {
            "sha256":
                closure_sha,
            "role":
                "HISTORICAL_PRE_REPAIR_ONLY",
        },

        "required_new_outputs": [
            "block48_formal_"
            "evaluation_repaired_v1.json",

            "stage4_final_"
            "closure_repaired_v1.json",

            "stage4_to_stage5_"
            "handoff_repaired_v1.json",
        ],

        "downstream_use_allowed":
            False,

        "formal_run_by_repair":
            False,

        "training_run_by_repair":
            False,

        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    write_json(
        STATUS,
        status,
    )

    print()
    print(
        "E. Syntax verification"
    )

    compile(
        MODULE.read_text(
            encoding="utf-8"
        ),
        str(MODULE),
        "exec",
    )

    compile(
        TEST.read_text(
            encoding="utf-8"
        ),
        str(TEST),
        "exec",
    )

    print(
        "syntax = PASS"
    )

    repair_report = {
        "stage": 4,

        "repair":
            "stage4_scientific_repair_v1",

        "status": "APPLIED",

        "Stage3":
            stage3,

        "historical":
            historical,

        "installed": {
            str(MODULE.relative_to(S4)):
                sha256(MODULE),

            str(TEST.relative_to(S4)):
                sha256(TEST),

            str(CONFIG.relative_to(S4)):
                sha256(CONFIG),

            str(STATUS.relative_to(S4)):
                sha256(STATUS),
        },

        "training_run":
            False,

        "formal_evaluation_run":
            False,

        "models_modified":
            False,

        "calibrator_modified":
            False,

        "next":
            "independent Stage4 "
            "repair checker",
    }

    write_json(
        REPORT,
        repair_report,
    )

    print()
    print(
        "=" * 60
    )

    print(
        "STAGE 4 SCIENTIFIC REPAIR V1 = APPLIED"
    )

    print(
        "=" * 60
    )

    print(
        "root   =",
        S4,
    )

    print(
        "backup =",
        BACKUP,
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "training run          = NO"
    )

    print(
        "formal evaluation run = NO"
    )

    print(
        "old Block4.8          = HISTORICAL / SUPERSEDED"
    )

    print(
        "next = independent Stage4 repair check"
    )


if __name__ == "__main__":

    try:
        main()

    except RepairError as exc:

        print()
        print(
            "=" * 60
        )

        print(
            "STAGE 4 SCIENTIFIC REPAIR V1 = ABORTED"
        )

        print(
            "=" * 60
        )

        print(
            "reason =",
            str(exc),
        )

        print(
            "No formal evaluation or training was run."
        )

        raise SystemExit(2)

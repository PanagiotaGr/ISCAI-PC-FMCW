from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def choose_probe_plan(
    formal_rows,
    cache_dir,
    *,
    count=4,
):
    """
    Deterministic preregistered probe rule:

      first `count` non-empty formal scenes
      in frozen manifest order.

    Selection uses only already-frozen cache
    metadata, never model performance.
    """

    cache_dir = Path(
        cache_dir
    )

    plan = []

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
                "Missing formal shard while "
                "constructing probe plan: "
                f"{path}"
            )

        with np.load(
            path,
            allow_pickle=False,
        ) as data:
            stored_id = str(
                data[
                    "scenario_id"
                ].item()
            )

            if (
                stored_id
                !=
                scenario_id
            ):
                raise RuntimeError(
                    "Formal shard scenario-ID "
                    "mismatch."
                )

            sample_count = int(
                data[
                    "all_supervised_prediction_count"
                ].item()
            )

            if sample_count <= 0:
                continue

            prediction_sha = str(
                data[
                    "all_prediction_sha256"
                ].item()
            )

            truth_index_source = str(
                data[
                    "truth_index_source"
                ].item()
            )

        if len(
            prediction_sha
        ) != 64:
            raise RuntimeError(
                "Invalid cached prediction SHA."
            )

        plan.append({
            "rank":
                int(
                    rank
                ),

            "scenario_id":
                scenario_id,

            "expected_prediction_sha256":
                prediction_sha,

            "expected_supervised_samples":
                sample_count,

            "truth_index_source":
                truth_index_source,
        })

        if (
            len(
                plan
            )
            ==
            int(
                count
            )
        ):
            break

    if (
        len(
            plan
        )
        !=
        int(
            count
        )
    ):
        raise RuntimeError(
            "Could not obtain requested number "
            "of non-empty formal probe scenes."
        )

    return plan


def probe_plan_sha256(
    plan,
):
    payload = json.dumps(
        plan,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        allow_nan=False,
    ).encode(
        "utf-8"
    )

    return sha256(
        payload
    ).hexdigest()


def aggregate_probe_results(
    plan,
    results,
):
    if (
        len(
            plan
        )
        !=
        len(
            results
        )
    ):
        raise RuntimeError(
            "Probe-plan/result length mismatch."
        )

    verified = []

    for expected, actual in zip(
        plan,
        results,
    ):
        if (
            actual.get(
                "status"
            )
            !=
            "PASS"
        ):
            raise RuntimeError(
                "Fresh-process probe "
                "is not PASS."
            )

        if (
            int(
                actual[
                    "rank"
                ]
            )
            !=
            int(
                expected[
                    "rank"
                ]
            )
        ):
            raise RuntimeError(
                "Fresh-process probe rank "
                "mismatch."
            )

        if (
            str(
                actual[
                    "scenario_id"
                ]
            )
            !=
            str(
                expected[
                    "scenario_id"
                ]
            )
        ):
            raise RuntimeError(
                "Fresh-process scenario "
                "mismatch."
            )

        if (
            str(
                actual[
                    "expected_prediction_sha256"
                ]
            )
            !=
            str(
                expected[
                    "expected_prediction_sha256"
                ]
            )
        ):
            raise RuntimeError(
                "Probe expected-SHA metadata "
                "mismatch."
            )

        if (
            str(
                actual[
                    "actual_prediction_sha256"
                ]
            )
            !=
            str(
                expected[
                    "expected_prediction_sha256"
                ]
            )
        ):
            raise RuntimeError(
                "Fresh-process prediction SHA "
                "does not match frozen Block4.8."
            )

        if (
            int(
                actual[
                    "supervised_samples"
                ]
            )
            !=
            int(
                expected[
                    "expected_supervised_samples"
                ]
            )
        ):
            raise RuntimeError(
                "Fresh-process supervised-sample "
                "count changed."
            )

        if not bool(
            actual.get(
                "strict_state_load"
            )
        ):
            raise RuntimeError(
                "Fresh process did not report "
                "strict checkpoint loading."
            )

        if not bool(
            actual.get(
                "exact_prediction_match"
            )
        ):
            raise RuntimeError(
                "Fresh-process exact prediction "
                "match flag is false."
            )

        if bool(
            actual.get(
                "tracks_to_predict_accessed"
            )
        ):
            raise RuntimeError(
                "Fresh reproducibility probe "
                "accessed tracks_to_predict."
            )

        verified.append({
            "rank":
                int(
                    expected[
                        "rank"
                    ]
                ),

            "scenario_id":
                str(
                    expected[
                        "scenario_id"
                    ]
                ),

            "prediction_sha256":
                str(
                    actual[
                        "actual_prediction_sha256"
                    ]
                ),

            "supervised_samples":
                int(
                    actual[
                        "supervised_samples"
                    ]
                ),
        })

    return {
        "probe_count":
            len(
                verified
            ),

        "all_exact":
            True,

        "verified":
            verified,
    }

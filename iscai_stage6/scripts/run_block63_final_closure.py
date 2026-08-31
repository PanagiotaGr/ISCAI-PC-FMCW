from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback
from typing import Any


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

SCENARIO_ID = "b85e1bd6cc8e74c0"

BINDING = (
    S6
    / "configs"
    / "block63_deterministic_predictive_binding.json"
)

PART1 = (
    S6
    / "reports"
    / "block63_part1_deterministic_future_box.json"
)

PART2A = (
    S6
    / "reports"
    / "block63_part2_real_causal_association.json"
)

PART2B1 = (
    S6
    / "reports"
    / "block63_part2b1_frozen_gaussian_forward.json"
)

PART2B2 = (
    S6
    / "reports"
    / "block63_part2b2_real_future_full_box.json"
)

PART2B2_ARTIFACT = (
    S6
    / "artifacts"
    / "block63"
    / "block63_part2b2_real_future_full_box.json"
)

POSTERIOR_JSONL = (
    S6
    / "artifacts"
    / "block63"
    / "identity_safe_development_gaussian_posterior.jsonl"
)

# Resolve exact NPZ through frozen SHA rather than assuming filename.
POSTERIOR_NPZ_ROOT = (
    S6
    / "artifacts"
    / "block63"
)

PART2B2_RUNNER = (
    S6
    / "scripts"
    / "run_block63_part2b2_real_future_full_box.py"
)

CLOSURE = (
    S6
    / "reports"
    / "block63_final_closure.json"
)

HANDOFF = (
    S6
    / "artifacts"
    / "block63"
    / "block63_to_block64_handoff.json"
)

FREEZE = (
    S6
    / "artifacts"
    / "block63"
    / "block63_final_freeze_manifest.json"
)


EXPECTED = {
    "binding":
        (
            "03903018109e7d5037643381ff71141a5c57f003"
            "b8bdc6b9be13bb4304468f67"
        ),

    "part1":
        (
            "871c8a81e43048ff2ccf7ef1844ad5433a859ff1"
            "5b944749e13be9160b08256a"
        ),

    "part2a":
        (
            "71c9d03cc9c1b33dbb371dccd7140d5226494554"
            "b6b21df7fbfa19e9f43ba71a"
        ),

    "part2b1":
        (
            "54a4c99394d0fe29e4e7e480deacaefd07682b4f"
            "a1f1f8d8555a051a5efdabaf"
        ),

    "part2b2":
        (
            "e747c4b75f89c1ac362a57eecebb54a970419ac9"
            "9a8eb101fb106b4927675c64"
        ),

    "part2b2_artifact":
        (
            "48118f193628937bf2daa94208ca801408aa76df"
            "b360465064884356d52e2c62"
        ),

    "part2b2_scientific_digest":
        (
            "dfb3e474e2eaebebd6f8d9b8f5292c12551de17"
            "c29dbd2c2820f4a5b06e6b86e"
        ),

    "posterior_jsonl":
        (
            "318b22dfc3788195a1432169075d1c8654f0d183"
            "337711918db2303b5668f5ea"
        ),

    "posterior_npz":
        (
            "2d35613bfb0bdc7133c5b3e96b4ab28e9cca643"
            "40442f9a2ab1bcf9db3aad914"
        ),

    "part2b2_runner":
        (
            "8ae81ac788de1e7880e15b620bc159f03de806e5"
            "8906d5789ec06fbb285d9dbc"
        ),
}


SOURCE_FILES = (
    S6
    / "src"
    / "iscai_stage6"
    / "adb"
    / "deterministic_association.py",

    S6
    / "src"
    / "iscai_stage6"
    / "adb"
    / "deterministic_predictive.py",

    S6
    / "src"
    / "iscai_stage6"
    / "adb"
    / "womd_geometry.py",

    S6
    / "src"
    / "iscai_stage6"
    / "adb"
    / "geometry.py",

    S6
    / "tests"
    / "test_block63_part1_deterministic_future_box.py",
)


class ClosureBlock(
    RuntimeError
):
    pass


def require(
    condition: Any,
    message: str,
) -> None:
    if not bool(condition):
        raise ClosureBlock(
            message
        )


def file_sha256(
    path: Path,
) -> str:
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


def read_json(
    path: Path,
) -> dict[str, Any]:
    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    require(
        isinstance(
            value,
            dict,
        ),
        f"{path} is not a JSON object.",
    )

    return value


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> str:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        +
        "\n",
        encoding="utf-8",
    )

    return file_sha256(
        path
    )


def find_by_sha(
    root: Path,
    *,
    suffix: str,
    expected_sha: str,
) -> Path:
    matches = []

    for path in root.rglob(
        f"*{suffix}"
    ):
        if not path.is_file():
            continue

        try:
            digest = file_sha256(
                path
            )
        except BaseException:
            continue

        if digest == expected_sha:
            matches.append(
                path
            )

    require(
        len(
            matches
        )
        ==
        1,
        (
            f"Expected exactly one {suffix} "
            "artifact with frozen SHA, found "
            f"{len(matches)}: {matches}"
        ),
    )

    return matches[
        0
    ]


def run_regression() -> dict[str, Any]:
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(
                S6
                / "tests"
            ),
            "-p",
            "test_*.py",
        ],
        cwd=str(
            S6
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        process.stdout,
    )

    count = (
        int(
            match.group(
                1
            )
        )
        if match
        else None
    )

    return {
        "returncode":
            int(
                process.returncode
            ),

        "tests":
            count,

        "tail":
            "\n".join(
                process.stdout
                .splitlines()[
                    -30:
                ]
            ),
    }


def main() -> None:
    try:
        print(
            "=" * 72
        )
        print(
            "BLOCK 6.3 — FINAL DETERMINISTIC PREDICTIVE BASELINE CLOSURE"
        )
        print(
            "=" * 72
        )

        # ====================================================
        # A. Exact prerequisite seal
        # ====================================================

        print()
        print(
            "===== A. EXACT BLOCK6.3 PREREQUISITE SEAL ====="
        )

        sealed = (
            (
                BINDING,
                EXPECTED[
                    "binding"
                ],
                "binding",
            ),
            (
                PART1,
                EXPECTED[
                    "part1"
                ],
                "Part1",
            ),
            (
                PART2A,
                EXPECTED[
                    "part2a"
                ],
                "Part2A",
            ),
            (
                PART2B1,
                EXPECTED[
                    "part2b1"
                ],
                "Part2B/1",
            ),
            (
                PART2B2,
                EXPECTED[
                    "part2b2"
                ],
                "Part2B/2",
            ),
            (
                PART2B2_ARTIFACT,
                EXPECTED[
                    "part2b2_artifact"
                ],
                "Part2B/2 artifact",
            ),
            (
                POSTERIOR_JSONL,
                EXPECTED[
                    "posterior_jsonl"
                ],
                "posterior JSONL",
            ),
            (
                PART2B2_RUNNER,
                EXPECTED[
                    "part2b2_runner"
                ],
                "Part2B/2 runner",
            ),
        )

        for path, expected_sha, label in sealed:
            require(
                path.is_file(),
                f"Missing {label}: {path}",
            )

            actual_sha = file_sha256(
                path
            )

            require(
                actual_sha
                ==
                expected_sha,
                (
                    f"{label} SHA changed. "
                    f"actual={actual_sha}"
                ),
            )

            print(
                f"{label:22s} = EXACT PASS"
            )

        posterior_npz = find_by_sha(
            POSTERIOR_NPZ_ROOT,
            suffix=".npz",
            expected_sha=(
                EXPECTED[
                    "posterior_npz"
                ]
            ),
        )

        print(
            "posterior NPZ          = EXACT PASS"
        )

        # ====================================================
        # B. Semantic closure
        # ====================================================

        print()
        print(
            "===== B. SCIENTIFIC SEMANTIC CLOSURE ====="
        )

        part2a = read_json(
            PART2A
        )

        part2b1 = read_json(
            PART2B1
        )

        part2b2 = read_json(
            PART2B2
        )

        execution_artifact = read_json(
            PART2B2_ARTIFACT
        )

        require(
            part2a.get(
                "status"
            )
            ==
            "PASS_REAL_CAUSAL_ASSOCIATION_EXECUTED",
            (
                "Part2A status changed."
            ),
        )

        require(
            part2b1.get(
                "status"
            )
            ==
            "PASS_IDENTITY_SAFE_FROZEN_GAUSSIAN_FORWARD",
            (
                "Part2B/1 status changed."
            ),
        )

        require(
            part2b2.get(
                "status"
            )
            ==
            "PASS_REAL_FUTURE_FULL_BOX_EXECUTED",
            (
                "Part2B/2 status changed."
            ),
        )

        require(
            part2b2.get(
                "scenario_id"
            )
            ==
            SCENARIO_ID,
            (
                "Part2B/2 scenario changed."
            ),
        )

        posterior = part2b2[
            "posterior"
        ]

        require(
            posterior[
                "records"
            ]
            ==
            27,
            (
                "Posterior record count changed."
            ),
        )

        require(
            posterior[
                "identity_safe"
            ]
            is True,
            (
                "Posterior is no longer "
                "identity-safe."
            ),
        )

        require(
            posterior[
                "prediction_id_retained"
            ]
            is True,
            (
                "prediction_id was not retained."
            ),
        )

        require(
            posterior[
                "causal_anchor_retained"
            ]
            is True,
            (
                "Causal anchor was not retained."
            ),
        )

        require(
            posterior[
                "anonymous_row_binding"
            ]
            is False,
            (
                "Anonymous posterior row "
                "binding reappeared."
            ),
        )

        require(
            posterior[
                "mean_only"
            ]
            is True,
            (
                "Block6.3 is no longer "
                "Gaussian-mean only."
            ),
        )

        require(
            posterior[
                "covariance_used"
            ]
            is False,
            (
                "Covariance entered deterministic "
                "Block6.3."
            ),
        )

        association = part2b2[
            "association"
        ]

        require(
            association
            ==
            {
                "matched":
                    6,

                "ADB_boxes":
                    9,

                "reactive_fallback":
                    3,

                "unmatched_predictions":
                    21,
            },
            (
                "Frozen association outcome changed: "
                f"{association}"
            ),
        )

        geometry = part2b2[
            "geometry"
        ]

        require(
            geometry[
                "future_boxes"
            ]
            ==
            24,
            (
                "Future box count changed."
            ),
        )

        require(
            geometry[
                "corners_per_box"
            ]
            ==
            8,
            (
                "Corner count per box changed."
            ),
        )

        require(
            geometry[
                "total_projected_corners"
            ]
            ==
            192,
            (
                "Projected corner total changed."
            ),
        )

        require(
            geometry[
                "full_box"
            ]
            is True,
            (
                "Full-box geometry disabled."
            ),
        )

        require(
            geometry[
                "centroid_only"
            ]
            is False,
            (
                "Centroid-only geometry reappeared."
            ),
        )

        require(
            tuple(
                float(x)
                for x in
                geometry[
                    "horizons_s"
                ]
            )
            ==
            (
                0.1,
                0.3,
                0.5,
                1.0,
            ),
            (
                "Frozen horizons changed."
            ),
        )

        provenance = part2b2[
            "provenance"
        ]

        require(
            provenance[
                "state_source_expected"
            ]
            ==
            "predicted_sample",
            (
                "Predictive state provenance "
                "changed."
            ),
        )

        require(
            provenance[
                "orientation_source_expected"
            ]
            ==
            "predicted_tangent",
            (
                "Predictive orientation provenance "
                "changed."
            ),
        )

        require(
            provenance[
                "persisted_predicted_sample"
            ]
            is True,
            (
                "predicted_sample runtime "
                "provenance missing."
            ),
        )

        require(
            provenance[
                "persisted_predicted_tangent"
            ]
            is True,
            (
                "predicted_tangent runtime "
                "provenance missing."
            ),
        )

        require(
            provenance[
                "future_GT"
            ]
            is False,
            (
                "Future GT entered controller."
            ),
        )

        require(
            provenance[
                "perfect_actor_identity"
            ]
            is False,
            (
                "Perfect actor identity entered "
                "controller."
            ),
        )

        execution = part2b2[
            "scientific_execution"
        ]

        for field in (
            "new_model_forward",
            "training",
            "recalibration",
            "formal_evaluation",
            "parameter_tuning",
        ):
            require(
                execution[
                    field
                ]
                is False,
                (
                    f"Unexpected scientific "
                    f"execution flag {field}=True."
                ),
            )

        require(
            part2b2[
                "regression"
            ][
                "status"
            ]
            ==
            "PASS",
            (
                "Part2B/2 regression status "
                "changed."
            ),
        )

        require(
            part2b2[
                "regression"
            ][
                "tests"
            ]
            ==
            98,
            (
                "Part2B/2 regression count "
                "changed."
            ),
        )

        require(
            part2b2[
                "artifact"
            ][
                "sha256"
            ]
            ==
            EXPECTED[
                "part2b2_artifact"
            ],
            (
                "Part2B/2 artifact link changed."
            ),
        )

        require(
            part2b2[
                "artifact"
            ][
                "scientific_digest"
            ]
            ==
            EXPECTED[
                "part2b2_scientific_digest"
            ],
            (
                "Part2B/2 scientific digest "
                "changed."
            ),
        )

        require(
            execution_artifact[
                "scientific_digest"
            ]
            ==
            EXPECTED[
                "part2b2_scientific_digest"
            ],
            (
                "Execution artifact scientific "
                "digest changed."
            ),
        )

        require(
            execution_artifact[
                "association"
            ][
                "hard_distance_threshold"
            ]
            is None,
            (
                "Unexpected hard association "
                "threshold."
            ),
        )

        require(
            execution_artifact[
                "association"
            ][
                "covariance_gate"
            ]
            is None,
            (
                "Unexpected covariance "
                "association gate."
            ),
        )

        require(
            execution_artifact[
                "association"
            ][
                "class_gate"
            ]
            is None,
            (
                "Unexpected class "
                "association gate."
            ),
        )

        require(
            execution_artifact[
                "association"
            ][
                "perfect_identity"
            ]
            is False,
            (
                "Perfect identity entered "
                "association."
            ),
        )

        print(
            "identity-safe posterior     = PASS"
        )

        print(
            "Gaussian mean only          = PASS"
        )

        print(
            "association                 = 6 matched / 3 fallback"
        )

        print(
            "future boxes                = 24"
        )

        print(
            "full projected corners      = 192"
        )

        print(
            "predicted_sample provenance = PASS"
        )

        print(
            "predicted_tangent provenance= PASS"
        )

        print(
            "future GT / perfect ID      = NO / NO"
        )

        print(
            "covariance used in Block6.3 = NO"
        )

        # ====================================================
        # C. Fresh regression
        # ====================================================

        print()
        print(
            "===== C. FRESH STAGE6 REGRESSION ====="
        )

        regression = run_regression()

        print(
            regression[
                "tail"
            ]
        )

        require(
            regression[
                "returncode"
            ]
            ==
            0,
            (
                "Fresh Stage6 regression failed."
            ),
        )

        require(
            regression[
                "tests"
            ]
            ==
            98,
            (
                "Fresh Stage6 regression count "
                f"changed: {regression['tests']}"
            ),
        )

        print(
            "fresh regression = PASS | tests =",
            regression[
                "tests"
            ],
        )

        # ====================================================
        # D. Implementation freeze inventory
        # ====================================================

        print()
        print(
            "===== D. IMPLEMENTATION FREEZE INVENTORY ====="
        )

        implementation = {}

        for path in SOURCE_FILES:
            require(
                path.is_file(),
                (
                    f"Missing implementation "
                    f"source: {path}"
                ),
            )

            digest = file_sha256(
                path
            )

            implementation[
                str(
                    path
                )
            ] = digest

            print(
                path.name,
                "=",
                digest,
            )

        implementation[
            str(
                PART2B2_RUNNER
            )
        ] = EXPECTED[
            "part2b2_runner"
        ]

        # ====================================================
        # E. Block6.3 final closure
        # ====================================================

        print()
        print(
            "===== E. BLOCK6.3 FINAL CLOSURE ====="
        )

        closure_payload = {
            "stage":
                6,

            "block":
                "6.3",

            "status":
                (
                    "PASS_BLOCK63_DETERMINISTIC_"
                    "PREDICTIVE_BASELINE_COMPLETE"
                ),

            "scenario":
                {
                    "development_pilot":
                        SCENARIO_ID,

                    "formal_population_used":
                        False,
                },

            "trajectory_source":
                {
                    "family":
                        "calibrated_Gaussian_GRU",

                    "identity_safe":
                        True,

                    "posterior_records":
                        27,

                    "jsonl_path":
                        str(
                            POSTERIOR_JSONL
                        ),

                    "jsonl_sha256":
                        EXPECTED[
                            "posterior_jsonl"
                        ],

                    "npz_path":
                        str(
                            posterior_npz
                        ),

                    "npz_sha256":
                        EXPECTED[
                            "posterior_npz"
                        ],

                    "deterministic_signal":
                        "Gaussian_mean_only",

                    "predictive_covariance_available":
                        True,

                    "predictive_covariance_used":
                        False,
                },

            "causal_association":
                {
                    "policy":
                        (
                            "reciprocal_unique_nearest_"
                            "current_anchor_H0"
                        ),

                    "hard_distance_threshold":
                        None,

                    "covariance_gate":
                        None,

                    "class_gate":
                        None,

                    "perfect_actor_identity":
                        False,

                    "matched":
                        6,

                    "ADB_boxes":
                        9,

                    "reactive_fallback":
                        3,

                    "unmatched_predictions":
                        21,

                    "fallback_policy":
                        (
                            "ORIGINAL_REACTIVE_ADB_"
                            "FALLBACK"
                        ),
                },

            "future_geometry":
                {
                    "horizons_s":
                        [
                            0.1,
                            0.3,
                            0.5,
                            1.0,
                        ],

                    "future_boxes":
                        24,

                    "corners_per_box":
                        8,

                    "total_projected_corners":
                        192,

                    "centroid_only":
                        False,

                    "state_source":
                        "predicted_sample",

                    "orientation_source":
                        "predicted_tangent",

                    "dimensions":
                        (
                            "current_causal_LWH_held_"
                            "over_deterministic_horizons"
                        ),
                },

            "scope":
                {
                    "future_GT":
                        False,

                    "tracks_to_predict_controller_input":
                        False,

                    "objects_of_interest_controller_input":
                        False,

                    "truth_identity_controller_input":
                        False,

                    "predictive_covariance_used":
                        False,

                    "probabilistic_occupancy":
                        False,

                    "class_aware_policy":
                        False,

                    "temporal_smoothing":
                        False,

                    "actuation_rate_limits":
                        False,

                    "communication_codebook_reused":
                        False,

                    "formal_grid_frozen":
                        False,

                    "formal_evaluation":
                        False,

                    "training":
                        False,

                    "recalibration":
                        False,
                },

            "regression":
                {
                    "status":
                        "PASS",

                    "tests":
                        98,
                },

            "frozen_evidence":
                {
                    "binding_sha256":
                        EXPECTED[
                            "binding"
                        ],

                    "part1_sha256":
                        EXPECTED[
                            "part1"
                        ],

                    "part2a_sha256":
                        EXPECTED[
                            "part2a"
                        ],

                    "part2b1_sha256":
                        EXPECTED[
                            "part2b1"
                        ],

                    "part2b2_sha256":
                        EXPECTED[
                            "part2b2"
                        ],

                    "part2b2_artifact_sha256":
                        EXPECTED[
                            "part2b2_artifact"
                        ],

                    "part2b2_scientific_digest":
                        EXPECTED[
                            "part2b2_scientific_digest"
                        ],

                    "part2b2_runner_sha256":
                        EXPECTED[
                            "part2b2_runner"
                        ],
                },

            "implementation_sha256":
                implementation,

            "next":
                (
                    "BLOCK6.4_PROBABILISTIC_"
                    "FULL_BOX_OCCUPANCY"
                ),
        }

        closure_sha = write_json(
            CLOSURE,
            closure_payload,
        )

        print(
            "closure =",
            CLOSURE,
        )

        print(
            "closure SHA256 =",
            closure_sha,
        )

        # ====================================================
        # F. Explicit Block6.4 handoff
        # ====================================================

        print()
        print(
            "===== F. BLOCK6.3 → BLOCK6.4 HANDOFF ====="
        )

        handoff_payload = {
            "stage":
                6,

            "from_block":
                "6.3",

            "to_block":
                "6.4",

            "status":
                (
                    "PASS_READY_FOR_BLOCK64_"
                    "PROBABILISTIC_FULL_BOX_OCCUPANCY"
                ),

            "upstream_closure":
                {
                    "path":
                        str(
                            CLOSURE
                        ),

                    "sha256":
                        closure_sha,
                },

            "shared_posterior":
                {
                    "family":
                        "calibrated_Gaussian_GRU",

                    "records":
                        27,

                    "jsonl_path":
                        str(
                            POSTERIOR_JSONL
                        ),

                    "jsonl_sha256":
                        EXPECTED[
                            "posterior_jsonl"
                        ],

                    "npz_path":
                        str(
                            posterior_npz
                        ),

                    "npz_sha256":
                        EXPECTED[
                            "posterior_npz"
                        ],

                    "mean_key":
                        "mean_displacement_H0_m",

                    "causal_anchor_key":
                        "latest_position_H0_m",

                    "raw_covariance_key":
                        (
                            "raw_predictive_"
                            "covariance_H0_m2"
                        ),

                    "calibrated_covariance_key":
                        (
                            "calibrated_predictive_"
                            "covariance_H0_m2"
                        ),

                    "horizons_s":
                        [
                            0.1,
                            0.3,
                            0.5,
                            1.0,
                        ],

                    "future_ground_truth_as_controller_input":
                        False,

                    "tracks_to_predict_as_controller_input":
                        False,

                    "truth_identity_as_controller_input":
                        False,
                },

            "frozen_association":
                {
                    "policy":
                        (
                            "reciprocal_unique_nearest_"
                            "current_anchor_H0"
                        ),

                    "hard_threshold":
                        None,

                    "covariance_gate":
                        None,

                    "class_gate":
                        None,

                    "forced_assignment":
                        False,

                    "perfect_actor_identity":
                        False,

                    "unmatched_ADB_policy":
                        (
                            "ORIGINAL_REACTIVE_ADB_"
                            "FALLBACK"
                        ),

                    "development_pilot_result":
                        {
                            "ADB_boxes":
                                9,

                            "matched":
                                6,

                            "fallback":
                                3,

                            "unmatched_predictors":
                                21,
                        },
                },

            "frozen_full_box_geometry":
                {
                    "centroid_only":
                        False,

                    "corners_per_box":
                        8,

                    "state_provenance":
                        "predicted_sample",

                    "orientation_provenance":
                        "predicted_tangent",

                    "deterministic_dimensions":
                        (
                            "current_causal_LWH_held_"
                            "over_horizons"
                        ),

                    "communication_beam_actuator_reused_as_ADB":
                        False,
                },

            "Block64_required_transition":
                {
                    "deterministic_mean_baseline_frozen":
                        True,

                    "calibrated_predictive_covariance_available":
                        True,

                    "calibrated_predictive_covariance_must_now_be_used":
                        True,

                    "uncertainty_must_propagate_through_full_boxes":
                        True,

                    "centroid_only_probability_projection_forbidden":
                        True,

                    "probabilistic_occupancy_not_yet_implemented":
                        True,

                    "class_aware_policy_not_yet_implemented":
                        True,

                    "temporal_smoothing_not_yet_implemented":
                        True,

                    "actuation_rate_limits_not_yet_implemented":
                        True,

                    "formal_numeric_grid_not_yet_frozen":
                        True,

                    "MC_sample_count_not_yet_frozen":
                        True,

                    "occupancy_gamma_not_yet_frozen":
                        True,

                    "class_margins_not_yet_frozen":
                        True,

                    "dimming_floors_not_yet_frozen":
                        True,

                    "smoothing_parameters_not_yet_frozen":
                        True,

                    "rate_limit_parameters_not_yet_frozen":
                        True,

                    "development_before_formal":
                        True,

                    "formal_tuning_at_handoff":
                        False,
                },

            "scientific_execution_at_handoff":
                {
                    "new_model_forward":
                        False,

                    "training":
                        False,

                    "recalibration":
                        False,

                    "formal_evaluation":
                        False,

                    "parameter_tuning":
                        False,
                },
        }

        handoff_sha = write_json(
            HANDOFF,
            handoff_payload,
        )

        print(
            "handoff =",
            HANDOFF,
        )

        print(
            "handoff SHA256 =",
            handoff_sha,
        )

        # ====================================================
        # G. Freeze manifest
        # ====================================================

        print()
        print(
            "===== G. BLOCK6.3 FINAL FREEZE MANIFEST ====="
        )

        freeze_payload = {
            "stage":
                6,

            "block":
                "6.3",

            "status":
                "BLOCK63_FROZEN",

            "closure":
                {
                    "path":
                        str(
                            CLOSURE
                        ),

                    "sha256":
                        closure_sha,
                },

            "handoff":
                {
                    "path":
                        str(
                            HANDOFF
                        ),

                    "sha256":
                        handoff_sha,
                },

            "immutable_inputs":
                {
                    str(
                        BINDING
                    ):
                        EXPECTED[
                            "binding"
                        ],

                    str(
                        PART1
                    ):
                        EXPECTED[
                            "part1"
                        ],

                    str(
                        PART2A
                    ):
                        EXPECTED[
                            "part2a"
                        ],

                    str(
                        PART2B1
                    ):
                        EXPECTED[
                            "part2b1"
                        ],

                    str(
                        PART2B2
                    ):
                        EXPECTED[
                            "part2b2"
                        ],

                    str(
                        PART2B2_ARTIFACT
                    ):
                        EXPECTED[
                            "part2b2_artifact"
                        ],

                    str(
                        POSTERIOR_JSONL
                    ):
                        EXPECTED[
                            "posterior_jsonl"
                        ],

                    str(
                        posterior_npz
                    ):
                        EXPECTED[
                            "posterior_npz"
                        ],

                    str(
                        PART2B2_RUNNER
                    ):
                        EXPECTED[
                            "part2b2_runner"
                        ],
                },

            "implementation_sha256":
                implementation,

            "scientific_lock":
                {
                    "deterministic_predictive_baseline":
                        True,

                    "association_policy":
                        "FROZEN",

                    "Gaussian_mean_semantics":
                        "FROZEN",

                    "full_box_geometry":
                        "FROZEN",

                    "fallback_policy":
                        "FROZEN",

                    "predictive_covariance_in_Block63":
                        "FORBIDDEN",

                    "future_GT_controller_input":
                        "FORBIDDEN",

                    "perfect_identity_controller_input":
                        "FORBIDDEN",

                    "formal_tuning":
                        "NOT_PERFORMED",
                },
        }

        freeze_sha = write_json(
            FREEZE,
            freeze_payload,
        )

        print(
            "freeze manifest =",
            FREEZE,
        )

        print(
            "freeze SHA256 =",
            freeze_sha,
        )

        # ====================================================
        # H. Final
        # ====================================================

        print()
        print(
            "=" * 72
        )

        print(
            "BLOCK 6.3 — FINAL"
        )

        print(
            "=" * 72
        )

        print(
            "STATUS = "
            "PASS_BLOCK63_DETERMINISTIC_PREDICTIVE_BASELINE_FROZEN"
        )

        print(
            "posterior records        = 27"
        )

        print(
            "matched / fallback       = 6 / 3"
        )

        print(
            "future boxes             = 24"
        )

        print(
            "projected corners        = 192"
        )

        print(
            "Gaussian signal          = MEAN ONLY"
        )

        print(
            "predictive covariance    = RETAINED FOR BLOCK6.4"
        )

        print(
            "future GT                = NO"
        )

        print(
            "perfect actor identity   = NO"
        )

        print(
            "formal evaluation        = NO"
        )

        print(
            "training/recalibration   = NO / NO"
        )

        print(
            "fresh Stage6 regression  = PASS | tests = 98"
        )

        print(
            "closure SHA256           =",
            closure_sha,
        )

        print(
            "handoff SHA256           =",
            handoff_sha,
        )

        print(
            "freeze SHA256            =",
            freeze_sha,
        )

        print(
            "NEXT = BLOCK6.4 "
            "PROBABILISTIC FULL-BOX OCCUPANCY"
        )

        print(
            "terminal remains open = YES"
        )

        print(
            "=" * 72
        )

    except BaseException as exc:
        print()
        print(
            "=" * 72
        )

        print(
            "BLOCK6.3 FINAL CLOSURE — CONTROLLED BLOCK"
        )

        print(
            "=" * 72
        )

        print(
            "exception type =",
            type(
                exc
            ).__name__,
        )

        print(
            "exception      =",
            str(
                exc
            ),
        )

        print()

        traceback.print_exc(
            limit=18
        )

        print()

        print(
            "new model forward      = NO"
        )

        print(
            "training/recalibration = NO / NO"
        )

        print(
            "formal evaluation      = NO"
        )

        print(
            "upstream modification  = NO"
        )

        print(
            "terminal remains open  = YES"
        )

        print(
            "=" * 72
        )

    # Deliberately no sys.exit().


if __name__ == "__main__":
    main()

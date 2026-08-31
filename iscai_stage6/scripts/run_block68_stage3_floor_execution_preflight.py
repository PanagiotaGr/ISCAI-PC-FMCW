from __future__ import annotations

from hashlib import sha256
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

PREREG = (
    S6 / "configs/"
    "block66_part3b_class_aware_policy_preregistration.json"
)

STAGE1 = (
    S6 / "configs/"
    "stage6_development_stage1_class_gamma_freeze.json"
)

STAGE2 = (
    S6 / "configs/"
    "stage6_development_stage2_class_margin_freeze.json"
)

CLASS_POLICY = (
    S6 / "src/iscai_stage6/adb/"
    "class_aware_policy.py"
)

METRIC_SEMANTICS = (
    S6 / "src/iscai_stage6/adb/"
    "metric_semantics.py"
)

EVALUATOR_REFERENCE = (
    S6 / "src/iscai_stage6/adb/"
    "evaluator_reference.py"
)

EVALUATOR_CONTRACT = (
    S6 / "configs/"
    "stage6_evaluator_reference_contract.json"
)

ROAD_ROI = (
    S6 / "artifacts/block68/"
    "frozen_road_roi_definition.json"
)

VEHICLE_SURROGATE = (
    S6 / "artifacts/block66/"
    "block66_part3c_vehicle_surrogate_evidence.jsonl"
)

RECONCILIATION = (
    S6 / "reports/"
    "block68_preoutcome_metric_reconciliation.json"
)

OUTPUT = (
    S6 / "configs/"
    "stage6_development_stage3_floor_execution_binding.json"
)

REPORT = (
    S6 / "reports/"
    "block68_stage3_floor_execution_preflight.json"
)


EXPECTED = {
    "prereg":
        "5704494a89b4b3c7a3c19b0df8176c550c00795ed17cf2906c343f340461429e",

    "stage1":
        "a55f21589f55a75afcb660a8a7278e802dd9e102f4c75ca7863697359bbb2a87",

    "stage2":
        "b061541f0563181cd07a91107a211b3519c6fdcc2e08243171db5b3bd9393ae8",

    "class_policy":
        "b998f468b98c2770c48c84a2c0aaaa2177c2d84b8fc838e80fef9b9da1ebc14b",

    "evaluator_reference":
        "2015136fdcbfe3f837bcba5a2ee05578a4d8a01c4ecfd52e4dfac9f864349891",

    "evaluator_contract":
        "c9ff0390820a3c37702fb391ba603244fffcf7c03c476045b590aa483e89bd7e",

    "road_roi":
        "832f4bc3215fe585fee781080c98b2928157f26e8a66d22f428f15604afb75e0",

    "vehicle_surrogate":
        "bfe8eb6d8b605d58f9b99d8097744b71b99ecb9ed9fd169191222375f31bc826",

    "reconciliation":
        "077a89fa5163f8862a3a5b7489d98929d9ae3e160ed9e030eb75f42a91d48f9f",
}

EXPECTED_TESTS = 224

CLASS_ORDER = (
    "TYPE_VEHICLE",
    "TYPE_PEDESTRIAN",
    "TYPE_CYCLIST",
)

EXPECTED_RISK_VECTOR = [
    "vehicle_shadow_zone_violation",
    "glare_risk_exposure",
    "1-pedestrian_visibility_proxy",
    "1-cyclist_visibility_proxy",
    "1-road_illumination_retention",
]

EXPECTED_LEXICOGRAPHIC = [
    "minimize maximum element of normalized risk vector",
    "minimize arithmetic mean of normalized risk vector",
    "minimize over_masking_area",
    "minimize false_dimming",
    "canonical floor tuple (f_vehicle,f_pedestrian,f_cyclist)",
]

METRIC_FUNCTIONS = (
    "vehicle_shadow_zone_violation",
    "glare_risk_exposure",
    "pedestrian_visibility_proxy",
    "cyclist_visibility_proxy",
    "road_illumination_retention",
    "over_masking_area",
    "false_dimming",
)

EVALUATOR_EXPECTED = (
    "aggregate_class_future_region",
    "aggregate_constructed_oracle_supports",
    "frozen_road_roi",
    "validate_final_metric_illumination",
)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def file_sha(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def exact(path, expected, label):
    require(
        path.is_file(),
        f"Missing {label}: {path}",
    )

    actual = file_sha(path)

    require(
        actual == expected,
        (
            f"{label} SHA mismatch\n"
            f"expected={expected}\n"
            f"actual={actual}"
        ),
    )

    return actual


def read_json(path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def canonical_bytes(value):
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        +
        "\n"
    ).encode("utf-8")


def write_once(path, value):
    payload = canonical_bytes(value)

    if path.exists():
        require(
            path.read_bytes() == payload,
            (
                "Existing binding differs: "
                f"{path}"
            ),
        )
        return

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_bytes(payload)
    temporary.replace(path)


def regression():
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
        ],
        cwd=str(S6),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    count = None

    for line in process.stdout.splitlines():
        match = re.search(
            r"Ran\s+(\d+)\s+tests?",
            line,
        )

        if match:
            count = int(
                match.group(1)
            )

    require(
        process.returncode == 0,
        (
            "Stage6 regression failed:\n"
            +
            "\n".join(
                process.stdout.splitlines()[-100:]
            )
        ),
    )

    require(
        count == EXPECTED_TESTS,
        (
            f"Expected {EXPECTED_TESTS} tests; "
            f"got {count}."
        ),
    )

    return count


def count_jsonl(path):
    count = 0

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        for line in stream:
            if line.strip():
                json.loads(line)
                count += 1

    return count


def enumerate_floor_triplets():
    triplets = []

    for j_vehicle in range(
        0,
        21,
    ):
        for j_pedestrian in range(
            1,
            21,
        ):
            for j_cyclist in range(
                1,
                21,
            ):
                if (
                    j_vehicle
                    >
                    j_pedestrian
                ):
                    continue

                if (
                    j_vehicle
                    >
                    j_cyclist
                ):
                    continue

                triplets.append(
                    {
                        "j_vehicle":
                            j_vehicle,

                        "j_pedestrian":
                            j_pedestrian,

                        "j_cyclist":
                            j_cyclist,

                        "f_vehicle":
                            j_vehicle / 20.0,

                        "f_pedestrian":
                            j_pedestrian / 20.0,

                        "f_cyclist":
                            j_cyclist / 20.0,
                    }
                )

    return triplets


def public_callables(module):
    rows = []

    for name in sorted(
        dir(module)
    ):
        if name.startswith("_"):
            continue

        value = getattr(
            module,
            name
        )

        if not callable(value):
            continue

        try:
            signature = str(
                inspect.signature(
                    value
                )
            )

        except Exception:
            continue

        rows.append(
            {
                "name":
                    name,

                "signature":
                    signature,
            }
        )

    return rows


def source_excerpt(function):
    try:
        return inspect.getsource(
            function
        )
    except Exception:
        return (
            "<source unavailable>"
        )


def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.8"
    )
    print(
        "STAGE-3 CLASS-FLOOR EXECUTION PREFLIGHT"
    )
    print(
        "NO FLOOR OUTCOMES / NO FORMAL"
    )
    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # A. Exact frozen upstream
    # --------------------------------------------------------

    print()
    print(
        "===== A. EXACT FROZEN UPSTREAM ====="
    )

    seals = {}

    for key, path in (
        ("prereg", PREREG),
        ("stage1", STAGE1),
        ("stage2", STAGE2),
        ("class_policy", CLASS_POLICY),
        ("evaluator_reference", EVALUATOR_REFERENCE),
        ("evaluator_contract", EVALUATOR_CONTRACT),
        ("road_roi", ROAD_ROI),
        ("vehicle_surrogate", VEHICLE_SURROGATE),
        ("reconciliation", RECONCILIATION),
    ):
        seals[key] = exact(
            path,
            EXPECTED[key],
            key,
        )

        print(
            f"{key:22s} = EXACT PASS"
        )

    stage1 = read_json(
        STAGE1
    )

    stage2 = read_json(
        STAGE2
    )

    require(
        stage1.get(
            "status"
        )
        ==
        "FROZEN_STAGE1_CLASS_GAMMA_BEFORE_STAGE2_MARGIN_SELECTION",
        (
            "Stage-1 freeze status changed."
        ),
    )

    require(
        stage2.get(
            "status"
        )
        ==
        "FROZEN_STAGE2_CLASS_MARGIN_BEFORE_STAGE3_FLOOR_SELECTION",
        (
            "Stage-2 freeze status changed."
        ),
    )

    print(
        "Stage-1 gamma     = IMMUTABLE"
    )

    print(
        "Stage-2 margins   = IMMUTABLE"
    )

    print(
        "Stage-3 outcomes  = NOT COMPUTED"
    )

    print(
        "formal evaluation = NO"
    )

    # --------------------------------------------------------
    # B. Exact Stage-3 preregistration
    # --------------------------------------------------------

    print()
    print(
        "===== B. EXACT STAGE-3 PREREGISTRATION ====="
    )

    prereg = read_json(
        PREREG
    )

    stage3 = (
        prereg[
            "development_selection_protocol"
        ][
            "stage_3_class_floors"
        ]
    )

    floor_parameterization = (
        prereg[
            "class_intensity_floor_parameterization"
        ]
    )

    require(
        stage3[
            "vary"
        ]
        ==
        "class floor triplets only",
        (
            "Stage-3 vary contract changed."
        ),
    )

    require(
        stage3[
            "gamma_and_margin"
        ]
        ==
        "fixed from stages 1 and 2",
        (
            "Stage-3 gamma/margin freeze "
            "contract changed."
        ),
    )

    require(
        stage3[
            "arbitrary_weighted_sum"
        ]
        is False,
        (
            "Stage-3 weighted-sum prohibition "
            "changed."
        ),
    )

    require(
        stage3[
            "normalized_risk_vector"
        ]
        ==
        EXPECTED_RISK_VECTOR,
        (
            "Stage-3 risk vector changed."
        ),
    )

    require(
        stage3[
            "risk_vector_all_components_unit_interval"
        ]
        is True,
        (
            "Stage-3 unit-interval contract "
            "changed."
        ),
    )

    require(
        stage3[
            "lexicographic_selection"
        ]
        ==
        EXPECTED_LEXICOGRAPHIC,
        (
            "Stage-3 lexicographic selection "
            "changed."
        ),
    )

    print(
        "vary = CLASS FLOOR TRIPLETS ONLY"
    )

    print(
        "risk vector ="
    )

    for value in EXPECTED_RISK_VECTOR:
        print(
            " ",
            value,
        )

    print(
        "selection = LEXICOGRAPHIC"
    )

    print(
        "weighted sum = FORBIDDEN"
    )

    # --------------------------------------------------------
    # C. Floor candidate-space exact enumeration
    # --------------------------------------------------------

    print()
    print(
        "===== C. EXACT FLOOR CANDIDATE SPACE ====="
    )

    require(
        floor_parameterization[
            "representation"
        ]
        ==
        "f_c = j_c / 20",
        (
            "Floor representation changed."
        ),
    )

    require(
        floor_parameterization[
            "candidate_triplets"
        ]
        ==
        3270,
        (
            "Frozen floor triplet count "
            "changed."
        ),
    )

    require(
        floor_parameterization[
            "vehicle"
        ][
            "j_min"
        ]
        ==
        0,
        "Vehicle j_min changed.",
    )

    require(
        floor_parameterization[
            "vehicle"
        ][
            "j_max"
        ]
        ==
        20,
        "Vehicle j_max changed.",
    )

    for name in (
        "pedestrian",
        "cyclist",
    ):
        require(
            floor_parameterization[
                name
            ][
                "j_min"
            ]
            ==
            1,
            f"{name} j_min changed.",
        )

        require(
            floor_parameterization[
                name
            ][
                "j_max"
            ]
            ==
            20,
            f"{name} j_max changed.",
        )

        require(
            floor_parameterization[
                name
            ][
                "zero_permitted"
            ]
            is False,
            (
                f"{name} zero floor "
                "became permitted."
            ),
        )

    expected_constraints = [
        "f_vehicle <= f_pedestrian",
        "f_vehicle <= f_cyclist",
        "f_pedestrian > 0",
        "f_cyclist > 0",
    ]

    require(
        floor_parameterization[
            "structural_constraints"
        ]
        ==
        expected_constraints,
        (
            "Floor structural constraints "
            "changed."
        ),
    )

    triplets = enumerate_floor_triplets()

    require(
        len(triplets)
        ==
        3270,
        (
            "Independent floor enumeration "
            f"produced {len(triplets)}, "
            "not 3270."
        ),
    )

    print(
        "vehicle j      = 0..20"
    )

    print(
        "pedestrian j   = 1..20"
    )

    print(
        "cyclist j      = 1..20"
    )

    print(
        "admissible triplets = 3270 PASS"
    )

    # --------------------------------------------------------
    # D. Stage1 + Stage2 exact selected policy readback
    # --------------------------------------------------------

    print()
    print(
        "===== D. IMMUTABLE STAGE1/STAGE2 READBACK ====="
    )

    s1 = stage1[
        "selected"
    ]

    s2 = stage2[
        "selected"
    ]

    expected_k = {
        "TYPE_VEHICLE":
            980,

        "TYPE_PEDESTRIAN":
            499,

        "TYPE_CYCLIST":
            557,
    }

    for actor_class, k in (
        expected_k.items()
    ):
        require(
            int(
                s1[
                    actor_class
                ][
                    "k"
                ]
            )
            ==
            k,
            (
                f"Stage-1 k changed for "
                f"{actor_class}."
            ),
        )

    expected_margin = {
        "TYPE_VEHICLE":
            (
                0.0,
                0.5,
                0.05,
            ),

        "TYPE_PEDESTRIAN":
            (
                0.0,
                0.5,
                0.0,
            ),

        "TYPE_CYCLIST":
            (
                0.0,
                0.5,
                0.25,
            ),
    }

    for actor_class, expected_values in (
        expected_margin.items()
    ):
        actual = (
            float(
                s2[
                    actor_class
                ][
                    "base_margin_m"
                ]
            ),
            float(
                s2[
                    actor_class
                ][
                    "uncertainty_multiplier"
                ]
            ),
            float(
                s2[
                    actor_class
                ][
                    "motion_multiplier"
                ]
            ),
        )

        require(
            actual
            ==
            expected_values,
            (
                f"Stage-2 margin changed for "
                f"{actor_class}: {actual}"
            ),
        )

        print(
            actor_class,
            "k=",
            expected_k[
                actor_class
            ],
            "margin=",
            expected_values,
        )

    # --------------------------------------------------------
    # E. Stage-3 terminal illumination binding
    # --------------------------------------------------------

    print()
    print(
        "===== E. STAGE-3 TERMINAL ILLUMINATION SEMANTICS ====="
    )

    stage4 = (
        prereg[
            "development_selection_protocol"
        ][
            "stage_4_temporal_and_rate"
        ]
    )

    require(
        stage4[
            "all_class_parameters"
        ]
        ==
        "fixed from stages 1-3",
        (
            "Stage-4 dependency on frozen "
            "Stage1-3 policy changed."
        ),
    )

    require(
        stage4[
            "vary"
        ]
        ==
        (
            "smoothing time constant and "
            "finite dim/bright rate pair"
        ),
        (
            "Stage-4 ownership of temporal/rate "
            "parameters changed."
        ),
    )

    from iscai_stage6.adb.class_aware_policy import (
        apply_actuation_rate_limit,
        compose_class_aware_illumination,
        temporal_smooth_schedule,
    )

    compose_source = source_excerpt(
        compose_class_aware_illumination
    )

    require(
        "raw_class_aware_illumination"
        in
        compose_source,
        (
            "Class-aware raw output field "
            "not found."
        ),
    )

    # Critical sequential-selection binding:
    # Stage-3 may not depend on unselected Stage-4
    # smoothing/rate numerics.
    stage3_terminal = (
        "ClassAwareComposition."
        "raw_class_aware_illumination"
    )

    print(
        "Stage-3 terminal map =",
        stage3_terminal,
    )

    print(
        "temporal smoothing  = NOT APPLIED DURING STAGE-3 SELECTION"
    )

    print(
        "rate limiting       = NOT APPLIED DURING STAGE-3 SELECTION"
    )

    print(
        "reason              = OWNED BY STAGE-4, NOT YET SELECTED"
    )

    print()
    print(
        "compose signature =",
        inspect.signature(
            compose_class_aware_illumination
        ),
    )

    print(
        "smoothing signature =",
        inspect.signature(
            temporal_smooth_schedule
        ),
    )

    print(
        "rate-limit signature =",
        inspect.signature(
            apply_actuation_rate_limit
        ),
    )

    # --------------------------------------------------------
    # F. Authoritative metric API binding
    # --------------------------------------------------------

    print()
    print(
        "===== F. AUTHORITATIVE STAGE-3 METRIC APIs ====="
    )

    import iscai_stage6.adb.metric_semantics as metric_module

    metric_rows = []

    for name in METRIC_FUNCTIONS:
        require(
            hasattr(
                metric_module,
                name,
            ),
            (
                "Authoritative metric function "
                f"missing: {name}"
            ),
        )

        function = getattr(
            metric_module,
            name,
        )

        signature = str(
            inspect.signature(
                function
            )
        )

        metric_rows.append(
            {
                "name":
                    name,

                "signature":
                    signature,

                "source":
                    source_excerpt(
                        function
                    ),
            }
        )

        print()
        print(
            name,
            signature,
        )

    metric_sha = file_sha(
        METRIC_SEMANTICS
    )

    print()
    print(
        "metric_semantics.py SHA256 =",
        metric_sha,
    )

    # --------------------------------------------------------
    # G. Exact evaluator-reference API inventory
    # --------------------------------------------------------

    print()
    print(
        "===== G. EXACT EVALUATOR-REFERENCE APIs ====="
    )

    import iscai_stage6.adb.evaluator_reference as evaluator_module

    evaluator_rows = public_callables(
        evaluator_module
    )

    for row in evaluator_rows:
        print(
            row[
                "name"
            ],
            row[
                "signature"
            ],
        )

    evaluator_names = {
        row[
            "name"
        ]
        for row in evaluator_rows
    }

    for name in EVALUATOR_EXPECTED:
        require(
            name
            in
            evaluator_names,
            (
                "Expected evaluator helper "
                f"missing: {name}"
            ),
        )

    # Also surface geometry region APIs required
    # for the vehicle future glare surrogate.
    import iscai_stage6.adb.geometry as geometry_module

    geometry_rows = []

    for name in sorted(
        dir(
            geometry_module
        )
    ):
        lower = name.lower()

        if not any(
            token in lower
            for token in (
                "region",
                "fraction",
                "project",
            )
        ):
            continue

        value = getattr(
            geometry_module,
            name
        )

        if not callable(value):
            continue

        try:
            signature = str(
                inspect.signature(
                    value
                )
            )
        except Exception:
            continue

        geometry_rows.append(
            {
                "name":
                    name,

                "signature":
                    signature,
            }
        )

    print()
    print(
        "vehicle-region geometry candidates:"
    )

    for row in geometry_rows:
        print(
            row[
                "name"
            ],
            row[
                "signature"
            ],
        )

    # --------------------------------------------------------
    # H. Frozen evaluator artifacts
    # --------------------------------------------------------

    print()
    print(
        "===== H. FROZEN VEHICLE-SURROGATE / ROAD-ROI EVIDENCE ====="
    )

    vehicle_rows = count_jsonl(
        VEHICLE_SURROGATE
    )

    require(
        vehicle_rows
        ==
        719,
        (
            "Expected 719 vehicle-surrogate "
            f"records; got {vehicle_rows}."
        ),
    )

    road_roi_value = read_json(
        ROAD_ROI
    )

    evaluator_contract = read_json(
        EVALUATOR_CONTRACT
    )

    print(
        "vehicle surrogate records =",
        vehicle_rows,
    )

    print(
        "vehicle surrogate role    = FROZEN EVIDENCE"
    )

    print(
        "road ROI artifact         = EXACT PASS"
    )

    print(
        "evaluator contract        = EXACT PASS"
    )

    print()
    print(
        "road ROI top-level keys =",
        sorted(
            road_roi_value.keys()
        ),
    )

    print(
        "evaluator contract top-level keys =",
        sorted(
            evaluator_contract.keys()
        ),
    )

    # --------------------------------------------------------
    # I. Freeze Stage-3 execution binding
    # --------------------------------------------------------

    print()
    print(
        "===== I. WRITE-ONCE STAGE-3 EXECUTION BINDING ====="
    )

    binding = {
        "stage":
            6,

        "block":
            "6.8_stage3_floor_execution_preflight",

        "status":
            "FROZEN_STAGE3_FLOOR_EXECUTION_BEFORE_FLOOR_OUTCOMES",

        "stage1_stage2": {
            "gamma":
                "immutable",

            "class_margin":
                "immutable",
        },

        "floor_candidate_space": {
            "representation":
                "f_c=j_c/20",

            "candidate_triplets":
                3270,

            "vehicle_j":
                [0, 20],

            "pedestrian_j":
                [1, 20],

            "cyclist_j":
                [1, 20],

            "constraints":
                expected_constraints,
        },

        "selection": {
            "risk_vector":
                EXPECTED_RISK_VECTOR,

            "risk_components_are_already_normalized_unit_interval":
                True,

            "lexicographic_order":
                EXPECTED_LEXICOGRAPHIC,

            "weighted_sum":
                False,
        },

        "stage3_illumination": {
            "terminal_value":
                stage3_terminal,

            "temporal_smoothing_applied":
                False,

            "actuation_rate_limit_applied":
                False,

            "reason":
                (
                    "temporal/rate parameters belong "
                    "to Stage-4 and are not selected "
                    "until after Stage-3 floors freeze"
                ),
        },

        "metric_runtime": {
            "path":
                str(
                    METRIC_SEMANTICS
                ),

            "sha256":
                metric_sha,

            "functions":
                metric_rows,
        },

        "evaluator_runtime": {
            "path":
                str(
                    EVALUATOR_REFERENCE
                ),

            "sha256":
                EXPECTED[
                    "evaluator_reference"
                ],

            "public_callables":
                evaluator_rows,

            "geometry_region_callables":
                geometry_rows,
        },

        "frozen_artifacts": {
            "vehicle_surrogate_evidence": {
                "path":
                    str(
                        VEHICLE_SURROGATE
                    ),

                "sha256":
                    EXPECTED[
                        "vehicle_surrogate"
                    ],

                "records":
                    vehicle_rows,
            },

            "road_roi": {
                "path":
                    str(
                        ROAD_ROI
                    ),

                "sha256":
                    EXPECTED[
                        "road_roi"
                    ],
            },

            "evaluator_contract": {
                "path":
                    str(
                        EVALUATOR_CONTRACT
                    ),

                "sha256":
                    EXPECTED[
                        "evaluator_contract"
                    ],
            },
        },

        "scientific_boundary": {
            "Stage3_floor_metrics_computed":
                False,

            "floor_triplet_selected":
                False,

            "Stage1_modified":
                False,

            "Stage2_modified":
                False,

            "Stage4_parameters_selected":
                False,

            "primary_acceptance_tested":
                False,

            "formal_evaluation":
                False,
        },

        "seals":
            seals,
    }

    write_once(
        OUTPUT,
        binding,
    )

    binding_sha = file_sha(
        OUTPUT
    )

    print(
        "Stage-3 execution binding SHA256 =",
        binding_sha,
    )

    # --------------------------------------------------------
    # J. Regression / final boundary
    # --------------------------------------------------------

    print()
    print(
        "===== J. FINAL SCIENTIFIC BOUNDARY ====="
    )

    tests = regression()

    print(
        "Stage-3 floor outcomes    = NOT COMPUTED"
    )

    print(
        "selected floor tuple      = NONE"
    )

    print(
        "P_occ reopened            = NO"
    )

    print(
        "Stage-1 gamma modified    = NO"
    )

    print(
        "Stage-2 margins modified  = NO"
    )

    print(
        "Stage-4 temporal/rate     = NOT SELECTED"
    )

    print(
        "primary acceptance        = NOT TESTED"
    )

    print(
        "formal evaluation         = NO"
    )

    print(
        "Stage6 regression         =",
        f"{tests} / {tests} PASS",
    )

    report = {
        "stage":
            6,

        "block":
            "6.8_stage3_floor_execution_preflight",

        "status":
            "PASS_STAGE3_FLOOR_EXECUTION_BOUND",

        "binding": {
            "path":
                str(
                    OUTPUT
                ),

            "sha256":
                binding_sha,
        },

        "floor_candidate_triplets":
            len(
                triplets
            ),

        "Stage3_terminal_illumination":
            stage3_terminal,

        "metric_semantics_sha256":
            metric_sha,

        "vehicle_surrogate_records":
            vehicle_rows,

        "Stage6_regression":
            tests,

        "next":
            (
                "execute development-only 3270-triplet "
                "Stage-3 floor selection with Stage1/2 "
                "immutable and no Stage4 temporal/rate"
            ),
    }

    REPORT.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 STAGE-3 FLOOR EXECUTION PREFLIGHT — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "Stage-1 gamma             = IMMUTABLE"
    )

    print(
        "Stage-2 margins           = IMMUTABLE"
    )

    print(
        "floor candidate triplets  = 3270 EXACT"
    )

    print(
        "risk-vector components    = 5 EXACT"
    )

    print(
        "selection                 = LEXICOGRAPHIC"
    )

    print(
        "weighted objective        = FORBIDDEN"
    )

    print(
        "Stage-3 terminal map      = raw_class_aware_illumination"
    )

    print(
        "temporal smoothing        = STAGE-4 ONLY"
    )

    print(
        "rate limiting             = STAGE-4 ONLY"
    )

    print(
        "metric APIs               = BOUND"
    )

    print(
        "vehicle surrogate         = 719 RECORDS EXACT"
    )

    print(
        "road ROI                  = EXACT FROZEN"
    )

    print(
        "Stage-3 outcomes          = NOT COMPUTED"
    )

    print(
        "formal evaluation         = NO"
    )

    print(
        "Stage6 regression         =",
        f"{tests} / {tests} PASS",
    )

    print(
        "STATUS = "
        "PASS_STAGE3_FLOOR_EXECUTION_BOUND"
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "terminal remains open = YES"
    )


try:
    main()

except BaseException as exc:

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 STAGE-3 FLOOR EXECUTION PREFLIGHT = BLOCKED"
    )
    print(
        "============================================================"
    )

    print(
        "exception =",
        type(exc).__name__,
        str(exc),
    )

    print()
    traceback.print_exc()

    print()
    print(
        "Stage-1 gamma            = STILL FROZEN"
    )

    print(
        "Stage-2 margins          = STILL FROZEN"
    )

    print(
        "Stage-3 floor outcomes   = NOT COMPUTED"
    )

    print(
        "Stage-4 temporal/rate    = NOT SELECTED"
    )

    print(
        "primary acceptance       = NOT TESTED"
    )

    print(
        "formal evaluation        = NO"
    )

    print(
        "Do not start the 3270-triplet "
        "floor sweep until this binding passes."
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

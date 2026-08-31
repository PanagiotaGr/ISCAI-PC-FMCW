from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import traceback


S6 = Path("/home/agni/waymo/iscai_stage6")


# ============================================================
# EXACT FROZEN AUTHORITIES
# ============================================================

PRIMARY_REPORT = (
    S6
    / "reports/"
      "block68_primary_development_acceptance_gate.json"
)

PRIMARY_DETAIL = (
    S6
    / "artifacts/block68/primary_development_gate/"
      "block68_primary_development_acceptance_detail.json"
)

STAGE1_FREEZE = (
    S6
    / "configs/"
      "stage6_development_stage1_class_gamma_freeze.json"
)

STAGE2_FREEZE = (
    S6
    / "configs/"
      "stage6_development_stage2_class_margin_freeze.json"
)

STAGE3_FREEZE = (
    S6
    / "configs/"
      "stage6_development_stage3_class_floor_freeze.json"
)

STAGE4_FREEZE = (
    S6
    / "configs/"
      "stage6_development_stage4_temporal_rate_selection_freeze.json"
)

ACCEPTANCE_POLICY = (
    S6
    / "configs/"
      "stage6_preformal_primary_acceptance_policy.json"
)

REACTIVE_COMPARATOR = (
    S6
    / "artifacts/block68/primary_development_gate/"
      "block68_reactive_primary_comparator_augmented.jsonl"
)

REACTIVE_MATERIALIZATION = (
    S6
    / "reports/"
      "block68_reactive_primary_comparator_materialization.json"
)

SCHEMA_REPAIR = (
    S6
    / "reports/"
      "block68_primary_gate_reactive_schema_repair.json"
)


# ============================================================
# WRITE-ONCE FAILURE CLOSURE
# ============================================================

CLOSURE = (
    S6
    / "reports/"
      "block68_development_failure_closure.json"
)

FREEZE_MANIFEST = (
    S6
    / "artifacts/block68/primary_development_gate/"
      "block68_development_failure_freeze_manifest.json"
)


EXPECTED_SHA = {
    "primary_report":
        "e77dd8fb42e0d4641d1f2bf86d4cdb3aba2c2fb89a610181304556b468edda59",

    "primary_detail":
        "41ef7d6249bbb4efcef1eafa8f315e7bcb89fb193338cdd3f5139e585efb5bf9",

    "stage1":
        "a55f21589f55a75afcb660a8a7278e802dd9e102f4c75ca7863697359bbb2a87",

    "stage2":
        "b061541f0563181cd07a91107a211b3519c6fdcc2e08243171db5b3bd9393ae8",

    "stage3":
        "c8536a1a2f888ebc3310cb900626cf0c8c0c2ca3ce34c2d321120a269c548d00",

    "stage4":
        "780e134a66153c52783bc18d55f607c891b9a6983f8cfc8ac78e2e7367874885",

    "acceptance":
        "e103466b1a6eb62be4e6746029147cbda9671b05130acba5210a3ab7328e6b81",

    "reactive_comparator":
        "5925f1e02dc3cb9dbad42f40f200337f60ccb87f4c919c4dd7c628eb3725c2fa",

    "reactive_materialization":
        "b8a99017d7502dd704ed646fc45925bba8a22e135c502cea91f871541168a9c4",
}


EXPECTED_TESTS = 224
TOL = 1.0e-12


EXPECTED = {
    "vehicle_reactive":
        0.09481248124008006,

    "vehicle_predictive":
        0.013116180708780553,

    "overmask_reactive":
        0.18862460571658454,

    "overmask_predictive":
        0.26521748861081823,

    "pedestrian_reactive":
        0.5699538664317828,

    "pedestrian_predictive":
        0.8992267609505494,

    "cyclist_reactive":
        0.7388137207027013,

    "cyclist_predictive":
        0.8869782482483739,

    "overmask_allowed_delta":
        0.02,
}


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def file_sha(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def exact(path: Path, expected: str, label: str):
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
            f"actual  ={actual}"
        ),
    )

    return actual


def read_json(path: Path):
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
        + "\n"
    ).encode("utf-8")


def write_once(path: Path, value):
    payload = canonical_bytes(value)

    if path.exists():

        require(
            path.read_bytes() == payload,
            (
                "Existing frozen artifact differs: "
                f"{path}"
            ),
        )

        return "EXISTING_EXACT"

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_bytes(payload)
    temporary.replace(path)

    return "WRITTEN"


def near(actual, expected, label):
    actual = float(actual)
    expected = float(expected)

    delta = abs(
        actual - expected
    )

    require(
        delta <= TOL,
        (
            f"{label} mismatch\n"
            f"actual={actual!r}\n"
            f"expected={expected!r}\n"
            f"delta={delta!r}"
        ),
    )


def run_regression():
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
            f"observed {count}."
        ),
    )

    return count


def main():

    print("=" * 78)
    print("STAGE 6 — BLOCK 6.8")
    print("DEVELOPMENT ACCEPTANCE FAILURE CLOSURE")
    print("FREEZE NEGATIVE RESULT / NO RETUNING / NO FORMAL")
    print("=" * 78)

    # --------------------------------------------------------
    # A. Exact frozen failure boundary
    # --------------------------------------------------------

    print()
    print("===== A. EXACT FROZEN FAILURE BOUNDARY =====")

    paths = {
        "primary_report":
            PRIMARY_REPORT,

        "primary_detail":
            PRIMARY_DETAIL,

        "stage1":
            STAGE1_FREEZE,

        "stage2":
            STAGE2_FREEZE,

        "stage3":
            STAGE3_FREEZE,

        "stage4":
            STAGE4_FREEZE,

        "acceptance":
            ACCEPTANCE_POLICY,

        "reactive_comparator":
            REACTIVE_COMPARATOR,

        "reactive_materialization":
            REACTIVE_MATERIALIZATION,
    }

    seals = {}

    for key, path in paths.items():

        seals[key] = exact(
            path,
            EXPECTED_SHA[key],
            key,
        )

        print(
            f"{key:24s} = EXACT PASS"
        )

    require(
        SCHEMA_REPAIR.is_file(),
        (
            "Missing primary-gate schema repair "
            f"report: {SCHEMA_REPAIR}"
        ),
    )

    schema_repair = read_json(
        SCHEMA_REPAIR
    )

    require(
        schema_repair.get("status")
        ==
        "PASS_PRIMARY_GATE_REACTIVE_SCHEMA_SOURCE_REPAIRED",
        (
            "Reactive schema repair status "
            "is not PASS."
        ),
    )

    schema_repair_sha = file_sha(
        SCHEMA_REPAIR
    )

    print(
        "schema_repair            = STATUS + SHA SEALED"
    )

    print(
        "schema repair SHA        =",
        schema_repair_sha,
    )

    # --------------------------------------------------------
    # B. Verify exact primary result
    # --------------------------------------------------------

    print()
    print("===== B. PRIMARY GATE RESULT READBACK =====")

    report = read_json(
        PRIMARY_REPORT
    )

    detail = read_json(
        PRIMARY_DETAIL
    )

    require(
        report.get("status")
        ==
        "FAIL_PRIMARY_DEVELOPMENT_ACCEPTANCE_FROZEN",
        "Primary report is not frozen FAIL.",
    )

    require(
        report.get("overall_pass")
        is False,
        "Primary report unexpectedly says overall PASS.",
    )

    require(
        detail.get("status")
        ==
        "FAIL_PRIMARY_DEVELOPMENT_ACCEPTANCE_FROZEN",
        "Primary detail is not frozen FAIL.",
    )

    require(
        detail.get("overall_pass")
        is False,
        "Primary detail unexpectedly says PASS.",
    )

    require(
        detail[
            "scientific_boundary"
        ][
            "post_outcome_retuning"
        ]
        is False,
        "Primary detail permits post-outcome retuning.",
    )

    require(
        detail[
            "scientific_boundary"
        ][
            "formal_evaluation"
        ]
        is False,
        "Formal evaluation unexpectedly occurred.",
    )

    gates = detail[
        "gates"
    ]

    require(
        gates[
            "vehicle_strict_improvement"
        ][
            "pass"
        ]
        is True,
        "Vehicle gate is not PASS.",
    )

    require(
        gates[
            "overmask_noninferiority"
        ][
            "pass"
        ]
        is False,
        "Overmask gate is not FAIL.",
    )

    require(
        gates[
            "pedestrian_visibility_noninferiority"
        ][
            "pass"
        ]
        is True,
        "Pedestrian gate is not PASS.",
    )

    require(
        gates[
            "cyclist_visibility_noninferiority"
        ][
            "pass"
        ]
        is True,
        "Cyclist gate is not PASS.",
    )

    print(
        "vehicle gate             = PASS"
    )

    print(
        "overmask gate            = FAIL"
    )

    print(
        "pedestrian gate          = PASS"
    )

    print(
        "cyclist gate             = PASS"
    )

    print(
        "overall primary gate     = FAIL"
    )

    # --------------------------------------------------------
    # C. Exact numeric verification
    # --------------------------------------------------------

    print()
    print("===== C. EXACT FAILURE NUMERICS =====")

    metrics = detail[
        "metrics"
    ]

    reactive = metrics[
        "reactive"
    ]

    predictive = metrics[
        "predictive"
    ]

    near(
        reactive[
            "vehicle_shadow_zone_violation"
        ],
        EXPECTED[
            "vehicle_reactive"
        ],
        "vehicle reactive",
    )

    near(
        predictive[
            "vehicle_shadow_zone_violation"
        ],
        EXPECTED[
            "vehicle_predictive"
        ],
        "vehicle predictive",
    )

    near(
        reactive[
            "over_masking_area"
        ],
        EXPECTED[
            "overmask_reactive"
        ],
        "overmask reactive",
    )

    near(
        predictive[
            "over_masking_area"
        ],
        EXPECTED[
            "overmask_predictive"
        ],
        "overmask predictive",
    )

    near(
        reactive[
            "pedestrian_visibility_proxy"
        ],
        EXPECTED[
            "pedestrian_reactive"
        ],
        "pedestrian reactive",
    )

    near(
        predictive[
            "pedestrian_visibility_proxy"
        ],
        EXPECTED[
            "pedestrian_predictive"
        ],
        "pedestrian predictive",
    )

    near(
        reactive[
            "cyclist_visibility_proxy"
        ],
        EXPECTED[
            "cyclist_reactive"
        ],
        "cyclist reactive",
    )

    near(
        predictive[
            "cyclist_visibility_proxy"
        ],
        EXPECTED[
            "cyclist_predictive"
        ],
        "cyclist predictive",
    )

    vehicle_delta = (
        predictive[
            "vehicle_shadow_zone_violation"
        ]
        -
        reactive[
            "vehicle_shadow_zone_violation"
        ]
    )

    overmask_delta = (
        predictive[
            "over_masking_area"
        ]
        -
        reactive[
            "over_masking_area"
        ]
    )

    overmask_ceiling = (
        reactive[
            "over_masking_area"
        ]
        +
        EXPECTED[
            "overmask_allowed_delta"
        ]
    )

    overmask_excess = (
        overmask_delta
        -
        EXPECTED[
            "overmask_allowed_delta"
        ]
    )

    pedestrian_delta = (
        predictive[
            "pedestrian_visibility_proxy"
        ]
        -
        reactive[
            "pedestrian_visibility_proxy"
        ]
    )

    cyclist_delta = (
        predictive[
            "cyclist_visibility_proxy"
        ]
        -
        reactive[
            "cyclist_visibility_proxy"
        ]
    )

    require(
        overmask_excess > 0.0,
        (
            "Expected positive overmask excess "
            "beyond the frozen bound."
        ),
    )

    print(
        "vehicle delta P-R       =",
        repr(
            vehicle_delta
        ),
    )

    print(
        "overmask reactive       =",
        repr(
            reactive[
                "over_masking_area"
            ]
        ),
    )

    print(
        "overmask predictive     =",
        repr(
            predictive[
                "over_masking_area"
            ]
        ),
    )

    print(
        "overmask delta P-R      =",
        repr(
            overmask_delta
        ),
    )

    print(
        "allowed overmask delta  =",
        EXPECTED[
            "overmask_allowed_delta"
        ],
    )

    print(
        "allowed predictive max  =",
        repr(
            overmask_ceiling
        ),
    )

    print(
        "excess beyond bound     =",
        repr(
            overmask_excess
        ),
    )

    print(
        "ped visibility delta    =",
        repr(
            pedestrian_delta
        ),
    )

    print(
        "cyclist visibility delta=",
        repr(
            cyclist_delta
        ),
    )

    # --------------------------------------------------------
    # D. Scientific interpretation
    # --------------------------------------------------------

    print()
    print("===== D. SCIENTIFIC COMPLETION STATUS =====")

    print(
        "vehicle protection improvement = YES"
    )

    print(
        "VRU visibility loss            = NO"
    )

    print(
        "over-masking controlled         = NO"
    )

    print(
        "Stage-6 completion gate         = NOT MET"
    )

    print(
        "formal Stage-6 evaluation       = BLOCKED BY FROZEN PROTOCOL"
    )

    print(
        "Stage-7 handoff                 = NOT ALLOWED"
    )

    print(
        "post-outcome tuning             = FORBIDDEN"
    )

    # --------------------------------------------------------
    # E. Regression
    # --------------------------------------------------------

    print()
    print("===== E. FULL STAGE6 REGRESSION =====")

    tests = run_regression()

    print(
        "Stage6 regression =",
        f"{tests} / {tests} PASS",
    )

    # --------------------------------------------------------
    # F. Write failure closure
    # --------------------------------------------------------

    print()
    print("===== F. WRITE-ONCE FAILURE CLOSURE =====")

    closure_payload = {
        "stage":
            6,

        "block":
            "6.8_development_failure_closure",

        "status":
            "STAGE6_DEVELOPMENT_ACCEPTANCE_FAILED_FROZEN",

        "complete_frozen":
            False,

        "Stage6_completion_criterion_met":
            False,

        "primary_development_acceptance": {
            "status":
                "FAIL_PRIMARY_DEVELOPMENT_ACCEPTANCE_FROZEN",

            "overall_pass":
                False,

            "vehicle_strict_improvement":
                True,

            "overmask_noninferiority":
                False,

            "pedestrian_visibility_noninferiority":
                True,

            "cyclist_visibility_noninferiority":
                True,
        },

        "failure_reason": {
            "metric":
                "over_masking_area",

            "reactive":
                float(
                    reactive[
                        "over_masking_area"
                    ]
                ),

            "predictive":
                float(
                    predictive[
                        "over_masking_area"
                    ]
                ),

            "predictive_minus_reactive":
                float(
                    overmask_delta
                ),

            "maximum_allowed_delta":
                0.02,

            "maximum_allowed_predictive":
                float(
                    overmask_ceiling
                ),

            "excess_beyond_allowed_delta":
                float(
                    overmask_excess
                ),

            "interpretation":
                (
                    "predictive ADB improves vehicle shadow-zone "
                    "protection and VRU visibility but violates "
                    "the frozen over-masking non-inferiority bound"
                ),
        },

        "frozen_policy": {
            "Stage1_gamma":
                "IMMUTABLE",

            "Stage2_margins":
                "IMMUTABLE",

            "Stage3_floors":
                {
                    "vehicle":
                        0.15,

                    "pedestrian":
                        1.0,

                    "cyclist":
                        1.0,
                },

            "Stage4_temporal_rate":
                {
                    "time_constant_ms":
                        200,

                    "rho_dim_per_s":
                        1,

                    "rho_bright_per_s":
                        1,
                },

            "post_outcome_retuning_allowed":
                False,
        },

        "protocol_boundary": {
            "formal_evaluation_run":
                False,

            "formal_evaluation_allowed_after_failure":
                False,

            "Stage7_handoff_allowed":
                False,

            "current_protocol_may_be_retuned":
                False,

            "negative_result_preserved":
                True,
        },

        "scientific_result": {
            "vehicle_shadow_zone_violation": {
                "reactive":
                    float(
                        reactive[
                            "vehicle_shadow_zone_violation"
                        ]
                    ),

                "predictive":
                    float(
                        predictive[
                            "vehicle_shadow_zone_violation"
                        ]
                    ),

                "delta":
                    float(
                        vehicle_delta
                    ),
            },

            "over_masking_area": {
                "reactive":
                    float(
                        reactive[
                            "over_masking_area"
                        ]
                    ),

                "predictive":
                    float(
                        predictive[
                            "over_masking_area"
                        ]
                    ),

                "delta":
                    float(
                        overmask_delta
                    ),
            },

            "pedestrian_visibility_proxy": {
                "reactive":
                    float(
                        reactive[
                            "pedestrian_visibility_proxy"
                        ]
                    ),

                "predictive":
                    float(
                        predictive[
                            "pedestrian_visibility_proxy"
                        ]
                    ),

                "delta":
                    float(
                        pedestrian_delta
                    ),
            },

            "cyclist_visibility_proxy": {
                "reactive":
                    float(
                        reactive[
                            "cyclist_visibility_proxy"
                        ]
                    ),

                "predictive":
                    float(
                        predictive[
                            "cyclist_visibility_proxy"
                        ]
                    ),

                "delta":
                    float(
                        cyclist_delta
                    ),
            },
        },

        "upstream_seals": {
            **seals,

            "schema_repair":
                schema_repair_sha,
        },

        "primary_result": {
            "report":
                str(
                    PRIMARY_REPORT
                ),

            "report_sha256":
                EXPECTED_SHA[
                    "primary_report"
                ],

            "detail":
                str(
                    PRIMARY_DETAIL
                ),

            "detail_sha256":
                EXPECTED_SHA[
                    "primary_detail"
                ],
        },

        "Stage6_regression":
            tests,

        "next":
            "STOP_CURRENT_STAGE6_PROTOCOL_NO_FORMAL_NO_STAGE7",
    }

    closure_state = write_once(
        CLOSURE,
        closure_payload,
    )

    closure_sha = file_sha(
        CLOSURE
    )

    print(
        "closure state =",
        closure_state,
    )

    print(
        "closure SHA   =",
        closure_sha,
    )

    # --------------------------------------------------------
    # G. Failure freeze manifest
    # --------------------------------------------------------

    print()
    print("===== G. WRITE-ONCE FAILURE FREEZE MANIFEST =====")

    manifest_payload = {
        "stage":
            6,

        "status":
            "FROZEN_NEGATIVE_DEVELOPMENT_RESULT",

        "Stage6_COMPLETE_FROZEN":
            False,

        "Stage6_DEVELOPMENT_FAILED_FROZEN":
            True,

        "Block6_9_formal_allowed":
            False,

        "Block6_10_success_closure_allowed":
            False,

        "Stage7_handoff_allowed":
            False,

        "current_protocol_retuning_allowed":
            False,

        "failure_closure": {
            "path":
                str(
                    CLOSURE
                ),

            "sha256":
                closure_sha,
        },

        "immutable_policy_seals": {
            "Stage1":
                EXPECTED_SHA[
                    "stage1"
                ],

            "Stage2":
                EXPECTED_SHA[
                    "stage2"
                ],

            "Stage3":
                EXPECTED_SHA[
                    "stage3"
                ],

            "Stage4":
                EXPECTED_SHA[
                    "stage4"
                ],

            "acceptance_policy":
                EXPECTED_SHA[
                    "acceptance"
                ],
        },

        "primary_failure_seals": {
            "report":
                EXPECTED_SHA[
                    "primary_report"
                ],

            "detail":
                EXPECTED_SHA[
                    "primary_detail"
                ],
        },

        "scientific_failure":
            "OVERMASK_NONINFERIORITY_BOUND_EXCEEDED",

        "negative_result_is_valid":
            True,

        "formal_evaluation_run":
            False,

        "Stage6_regression":
            tests,
    }

    manifest_state = write_once(
        FREEZE_MANIFEST,
        manifest_payload,
    )

    manifest_sha = file_sha(
        FREEZE_MANIFEST
    )

    print(
        "manifest state =",
        manifest_state,
    )

    print(
        "manifest SHA   =",
        manifest_sha,
    )

    # --------------------------------------------------------
    # H. Final
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("BLOCK 6.8 DEVELOPMENT FAILURE CLOSURE — FINAL")
    print("=" * 78)

    print(
        "vehicle protection             = PASS"
    )

    print(
        "pedestrian visibility          = PASS"
    )

    print(
        "cyclist visibility             = PASS"
    )

    print(
        "overmask noninferiority        = FAIL"
    )

    print(
        "overmask delta                 =",
        repr(
            overmask_delta
        ),
    )

    print(
        "overmask allowed delta         = 0.02"
    )

    print(
        "overmask excess beyond bound   =",
        repr(
            overmask_excess
        ),
    )

    print(
        "primary development acceptance = FAIL / FROZEN"
    )

    print(
        "Stage-6 completion criterion   = NOT MET"
    )

    print(
        "Stage1/2/3/4 policy            = IMMUTABLE"
    )

    print(
        "post-outcome retuning          = FORBIDDEN"
    )

    print(
        "Block 6.9 formal evaluation    = NOT ALLOWED"
    )

    print(
        "Stage-7 handoff                = NOT ALLOWED"
    )

    print(
        "Stage6 regression              =",
        f"{tests} / {tests} PASS",
    )

    print(
        "failure closure SHA            =",
        closure_sha,
    )

    print(
        "failure freeze manifest SHA    =",
        manifest_sha,
    )

    print(
        "STATUS = "
        "STAGE6_DEVELOPMENT_ACCEPTANCE_FAILED_FROZEN"
    )

    print(
        "NEXT = STOP CURRENT FROZEN STAGE6 PROTOCOL"
    )

    print(
        "terminal remains open = YES"
    )


try:
    main()

except BaseException as exc:

    print()
    print("=" * 78)
    print(
        "BLOCK 6.8 DEVELOPMENT FAILURE CLOSURE = BLOCKED"
    )
    print("=" * 78)

    print(
        "exception =",
        type(exc).__name__,
        str(exc),
    )

    print()
    traceback.print_exc()

    print()
    print(
        "PRIMARY SCIENTIFIC RESULT REMAINS:"
    )

    print(
        "FAIL_PRIMARY_DEVELOPMENT_ACCEPTANCE_FROZEN"
    )

    print(
        "Stage1/2/3/4 policy = IMMUTABLE"
    )

    print(
        "formal evaluation   = NO"
    )

    print(
        "post-outcome tuning = FORBIDDEN"
    )

    print(
        "Do not reinterpret a closure-script failure "
        "as permission to rerun development selection."
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

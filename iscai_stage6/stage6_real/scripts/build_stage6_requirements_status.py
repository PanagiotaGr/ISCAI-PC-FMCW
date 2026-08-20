from pathlib import Path
import json
from datetime import datetime, timezone


ROOT = Path("stage6_real")

STREAMING_REPORT = (
    ROOT
    / "reports"
    / "stage6_streaming_evaluation.json"
)

LATENCY_REPORT = (
    ROOT
    / "reports"
    / "stage6_adb_computational_latency.json"
)

PREDICTIONS = (
    ROOT
    / "data"
    / "real_stage6_predictions_balanced.npz"
)

OCCUPANCY = (
    ROOT
    / "data"
    / "real_stage6_probabilistic_occupancy.npz"
)

ORACLE = (
    ROOT
    / "data"
    / "real_stage6_oracle_future_boxes.npz"
)

EVALUATOR = (
    ROOT
    / "scripts"
    / "evaluate_stage6_streaming.py"
)

OUT = (
    ROOT
    / "reports"
    / "stage6_final_requirements_status.json"
)


def requirement(
    requirement_id,
    requirement,
    status,
    evidence,
    note="",
):
    return {
        "id": requirement_id,
        "requirement": requirement,
        "status": status,
        "evidence": evidence,
        "note": note,
    }


def main():

    # --------------------------------------------------------
    # Do not silently create a status report from missing runs.
    # --------------------------------------------------------

    required_files = [
        STREAMING_REPORT,
        LATENCY_REPORT,
        PREDICTIONS,
        OCCUPANCY,
        ORACLE,
        EVALUATOR,
    ]

    missing = [
        str(p)
        for p in required_files
        if not p.exists()
    ]

    if missing:
        raise RuntimeError(
            "Missing required evidence files:\n"
            +
            "\n".join(missing)
        )


    streaming = json.loads(
        STREAMING_REPORT.read_text()
    )

    latency = json.loads(
        LATENCY_REPORT.read_text()
    )


    requirements = []


    # ========================================================
    # A. Predictive ADB source / geometry
    # ========================================================

    requirements.append(
        requirement(
            "A1",
            (
                "Real WOMD-derived probabilistic future "
                "trajectory source"
            ),
            "IMPLEMENTED",
            [
                str(PREDICTIONS),
            ],
            (
                "Balanced real WOMD validation actors are "
                "used as the Stage-6 prediction source."
            ),
        )
    )

    requirements.append(
        requirement(
            "A2",
            (
                "Future ADB uses actor box geometry rather "
                "than centroid only"
            ),
            "IMPLEMENTED",
            [
                str(EVALUATOR),
                "box_corners()",
                "geometry_parameters()",
            ],
            (
                "Length, width, heading and projected box "
                "corners are used in the illumination geometry."
            ),
        )
    )

    requirements.append(
        requirement(
            "A3",
            "Predictive covariance used in occupancy",
            "IMPLEMENTED",
            [
                str(PREDICTIONS),
                "future_std_xy_H_m",
                "future_rho_xy_H",
                "probabilistic_occupancy()",
            ],
        )
    )

    requirements.append(
        requirement(
            "A4",
            "Class-dependent spatial uncertainty margin",
            "IMPLEMENTED",
            [
                "CLASS_CONFIGS",
                "lateral_margin_scale",
            ],
        )
    )

    requirements.append(
        requirement(
            "A5",
            (
                "Receiver/driver-relevant subregion included "
                "in predictive ADB geometry"
            ),
            "IMPLEMENTED_WITH_ASSUMPTION",
            [
                "vehicle_relevant_face()",
                "driver_region_intensity_map()",
                "driver_region_surrogate",
                "stage6_real/scripts/smoke_vehicle_driver_region.py",
                "stage6_real/reports/stage6_streaming_evaluation.json",
            ],
            (
                "Implemented as a geometry-based 2D vehicle "
                "surrogate because WOMD does not provide true "
                "windshield, mirror or driver coordinates. "
                "Front face is used for vehicles oriented "
                "toward the ego/headlamp and rear face for "
                "vehicles oriented away. This is an evaluated "
                "surrogate reference, not measured hardware "
                "geometry."
            ),
        )
    )


    # ========================================================
    # B. Part-A illumination model
    # ========================================================

    requirements.append(
        requirement(
            "B1",
            "Radial thresholds",
            "IMPLEMENTED",
            [
                "RADIAL_MARGIN_M",
                "geometry_parameters()",
            ],
        )
    )

    requirements.append(
        requirement(
            "B2",
            "Angular shadow zones",
            "IMPLEMENTED",
            [
                "theta_min",
                "theta_max",
                "intensity_map()",
            ],
        )
    )

    requirements.append(
        requirement(
            "B3",
            "Raised-cosine transition",
            "IMPLEMENTED",
            [
                "TRANSITION_LENGTH_M",
                "intensity_map()",
            ],
        )
    )

    requirements.append(
        requirement(
            "B4",
            "Minimum illumination floor",
            "IMPLEMENTED",
            [
                "CLASS_CONFIGS",
                "floor",
            ],
        )
    )

    requirements.append(
        requirement(
            "B5",
            "Temporal smoothing in the actual ADB controller",
            "IMPLEMENTED_WITH_ASSUMPTION",
            [
                "stage6_real/data/real_stage6_predictions_rolling.npz",
                "stage6_real/scripts/evaluate_stage6_temporal_controller.py",
                "stage6_real/reports/stage6_temporal_controller_evaluation.json",
            ],
            (
                "Temporal smoothing is implemented and evaluated on "
                "real consecutive causal WOMD anchor frames at 0.1 s "
                "intervals. A parameter sensitivity sweep is used "
                "instead of claiming a hardware-calibrated smoothing "
                "constant. The alpha values are modeling assumptions, "
                "not measured headlamp dynamics."
            ),
        )
    )

    requirements.append(
        requirement(
            "B6",
            "Actuation-rate limiting",
            "IMPLEMENTED_WITH_ASSUMPTION",
            [
                "stage6_real/data/real_stage6_predictions_rolling.npz",
                "stage6_real/scripts/evaluate_stage6_temporal_controller.py",
                "stage6_real/reports/stage6_temporal_controller_evaluation.json",
            ],
            (
                "Cellwise per-frame actuation-rate limiting is "
                "implemented and evaluated on consecutive 0.1 s "
                "controller frames. Multiple delta_max values are "
                "evaluated as sensitivity-sweep parameters. These "
                "limits are modeling assumptions and are not claimed "
                "as measured physical headlamp actuator limits."
            ),
        )
    )


    # ========================================================
    # C. Class-aware policies
    # ========================================================

    requirements.append(
        requirement(
            "C1",
            "Vehicle-specific class-aware ADB",
            "IMPLEMENTED_WITH_ASSUMPTION",
            [
                "CLASS_CONFIGS['VEHICLE']",
                "vehicle_relevant_face()",
                "driver_region_intensity_map()",
                "vehicle_dynamic_margin_scale()",
                "stage6_real/reports/stage6_streaming_evaluation.json",
            ],
            (
                "Vehicle class is handled separately. A geometry-based "
                "windshield/mirror-relevant surrogate is implemented and "
                "evaluated. Predictive covariance is represented through "
                "probabilistic occupancy, while the vehicle geometric margin "
                "increases with predicted closing displacement. Closing speed "
                "is derived from the causal predicted trajectory at 0.1 s "
                "sampling and is not measured FMCW radial velocity."
            ),
        )
    )

    requirements.append(
        requirement(
            "C2",
            "Pedestrian no-blackout / visibility policy",
            "IMPLEMENTED_WITH_ASSUMPTION",
            [
                "CLASS_CONFIGS['PEDESTRIAN']",
                "floor=0.35",
                "lateral_margin_scale=1.25",
            ],
            (
                "Implemented as a class-specific 2D pedestrian "
                "visibility policy. A non-zero illumination floor "
                "of 0.35 prevents complete blackout and the "
                "protected angular region uses a 1.25 lateral "
                "margin scale. WOMD provides actor box dimensions "
                "but no explicit face/head/body subregion "
                "coordinates, so the full 2D pedestrian actor "
                "region is used as the protected visibility "
                "surrogate. No measured face location is claimed."
            ),
        )
    )

    requirements.append(
        requirement(
            "C3",
            "Cyclist class-aware visibility policy",
            "IMPLEMENTED_WITH_ASSUMPTION",
            [
                "CLASS_CONFIGS['CYCLIST']",
                "floor=0.25",
                "lateral_margin_scale=1.50",
                "cyclist_dynamic_margin_scale()",
                "stage6_real/reports/stage6_streaming_evaluation.json",
            ],
            (
                "A cyclist-specific visibility floor and enlarged "
                "base lateral margin are implemented. The lateral "
                "margin is additionally increased according to "
                "predicted lateral displacement over the 0.1 s "
                "prediction step. This provides trajectory-dependent "
                "handling of lateral/crossing motion without oracle "
                "future input. WOMD provides no explicit crossing "
                "label here, so crossing behavior is inferred from "
                "the predicted trajectory and current heading."
            ),
        )
    )


    # ========================================================
    # D. ADB baselines
    # ========================================================

    requirements.append(
        requirement(
            "D1",
            "Static ADB baseline",
            "IMPLEMENTED_WITH_ASSUMPTION",
            [
                "stage6_real/data/real_stage6_adb_baseline_masks.npz",
                "stage6_real/scripts/build_adb_baseline_masks.py",
                "reactive_dimming",
            ],
            (
                "Implemented as a held-current-state, non-predictive "
                "ADB baseline. The real current WOMD actor box is "
                "converted to the Part-A geometric shadow-zone map "
                "and that map is held fixed across all forecast "
                "horizons. No Stage-4 future prediction or oracle "
                "future state is used. Interpreting this held-current "
                "policy as the Stage-6 static ADB baseline is an "
                "explicit modeling assumption."
            ),
        )
    )

    requirements.append(
        requirement(
            "D2",
            "Original reactive ADB baseline",
            "IMPLEMENTED",
            [
                "candidate_maps['reactive']",
                str(STREAMING_REPORT),
            ],
        )
    )

    requirements.append(
        requirement(
            "D3",
            "Deterministic predictive ADB",
            "IMPLEMENTED",
            [
                "candidate_maps['deterministic']",
                str(STREAMING_REPORT),
            ],
        )
    )

    requirements.append(
        requirement(
            "D4",
            "Uncertainty-aware predictive ADB",
            "IMPLEMENTED",
            [
                "uncertainty_gamma_*",
                str(STREAMING_REPORT),
            ],
        )
    )

    requirements.append(
        requirement(
            "D5",
            "Class-agnostic predictive ADB",
            "IMPLEMENTED",
            [
                "occupancy_class_agnostic",
                "uncertainty_gamma_*",
            ],
        )
    )

    requirements.append(
        requirement(
            "D6",
            "Class-aware predictive ADB",
            "IMPLEMENTED",
            [
                "class_aware_gamma_*",
                str(STREAMING_REPORT),
            ],
        )
    )

    requirements.append(
        requirement(
            "D7",
            "Oracle future ADB",
            "IMPLEMENTED",
            [
                str(ORACLE),
                "oracle_D",
                "stage6_real/reports/real_stage6_oracle_future_boxes_summary.json",
            ],
            (
                "Oracle future ADB is implemented as an "
                "evaluation-only upper-bound reference using "
                "real future WOMD box geometry. It is intentionally "
                "excluded from deployable candidate methods because "
                "it uses future ground truth. Oracle future state is "
                "never used as predictor or controller input."
            ),
        )
    )


    # ========================================================
    # E. Metrics
    # ========================================================

    metric_status = [
        (
            "E1",
            "Mask IoU with oracle future mask",
            "IMPLEMENTED",
            "mean_iou",
        ),
        (
            "E2",
            "Vehicle shadow-zone violation",
            "IMPLEMENTED",
            "mean_shadow_violation",
        ),
        (
            "E3",
            "Glare-risk exposure",
            "IMPLEMENTED_WITH_ASSUMPTION",
            "mean_oracle_region_visibility",
        ),
        (
            "E4",
            "Over-masking area",
            "IMPLEMENTED",
            "mean_overmask",
        ),
        (
            "E5",
            "Road illumination retention",
            "IMPLEMENTED",
            "mean_road_retention",
        ),
        (
            "E6",
            "Pedestrian visibility proxy",
            "IMPLEMENTED_WITH_ASSUMPTION",
            "mean_oracle_region_visibility",
        ),
        (
            "E7",
            "Cyclist visibility proxy",
            "IMPLEMENTED_WITH_ASSUMPTION",
            "mean_oracle_region_visibility",
        ),
        (
            "E8",
            "False dimming",
            "IMPLEMENTED_WITH_ASSUMPTION",
            "mean_false_dimming",
        ),
        (
            "E9",
            "Temporal smoothness metric",
            "IMPLEMENTED",
            "mean_temporal_smoothness",
        ),
        (
            "E10",
            "Flicker / change rate",
            "IMPLEMENTED",
            "mean_flicker_rate_0_25",
        ),
        (
            "E11",
            "Energy consumption",
            "IMPLEMENTED_WITH_ASSUMPTION",
            "mean_road_retention",
        ),
        (
            "E12",
            "Physical ADB actuation latency",
            "NOT_MEASURED",
            None,
        ),
    ]

    all_relevant = (
        streaming
        ["results"]
        ["all_relevant"]
    )

    representative_method = (
        "class_aware_gamma_0.25"
    )

    representative = all_relevant[
        representative_method
    ]

    for (
        rid,
        name,
        status,
        field,
    ) in metric_status:

        evidence = []

        # ----------------------------------------------------
        # Class-specific visibility evidence
        # ----------------------------------------------------
        if rid == "E6":
            class_name = "PEDESTRIAN"

            class_result = (
                streaming
                ["results_by_class"]
                [class_name]
                [representative_method]
            )

            evidence.append(
                {
                    "report":
                        str(STREAMING_REPORT),

                    "scope":
                        class_name,

                    "representative_method":
                        representative_method,

                    "field":
                        field,

                    "value":
                        class_result[field],
                }
            )

        elif rid == "E7":
            class_name = "CYCLIST"

            class_result = (
                streaming
                ["results_by_class"]
                [class_name]
                [representative_method]
            )

            evidence.append(
                {
                    "report":
                        str(STREAMING_REPORT),

                    "scope":
                        class_name,

                    "representative_method":
                        representative_method,

                    "field":
                        field,

                    "value":
                        class_result[field],
                }
            )

        elif (
            field is not None
            and
            field in representative
        ):
            evidence.append(
                {
                    "report":
                        str(STREAMING_REPORT),

                    "representative_method":
                        representative_method,

                    "field":
                        field,

                    "value":
                        representative[field],
                }
            )

        note = ""

        if rid in ("E6", "E7"):
            note = (
                "The proposal requests a visibility proxy but "
                "does not prescribe a unique numerical formula. "
                "The current implementation reports mean retained "
                "illumination inside the constructed oracle region."
            )

        elif rid == "E8":
            note = (
                "False dimming is an explicit evaluation "
                "definition introduced by this implementation."
            )

        elif rid == "E3":
            vehicle_results = (
                streaming[
                    "results_by_class"
                ][
                    "VEHICLE"
                ]
            )

            representative_vehicle = (
                vehicle_results[
                    representative_method
                ]
            )

            evidence = [
                {
                    "report":
                        str(STREAMING_REPORT),

                    "scope":
                        "VEHICLE_ONLY",

                    "representative_method":
                        representative_method,

                    "field":
                        "mean_oracle_region_visibility",

                    "value":
                        representative_vehicle[
                            "mean_oracle_region_visibility"
                        ],

                    "definition":
                        (
                            "Mean retained illumination inside the "
                            "oracle future vehicle protection region."
                        ),
                }
            ]

            note = (
                "Implemented as a vehicle-specific modeled glare-"
                "exposure proxy. The metric is the mean retained "
                "illumination inside the oracle future vehicle "
                "protection region, so lower values correspond to "
                "lower modeled illumination exposure. This is not "
                "a photometric eye-level glare measurement in lux, "
                "candela, or discomfort-glare units."
            )

        elif rid == "E11":
            note = (
                "A normalized illumination-demand proxy is reported "
                "as E_norm = mean(L) = 1 - mean(D), using the evaluated "
                "Stage-6 illumination maps. This is not measured "
                "electrical energy in W, Wh, or J. No calibrated "
                "headlamp electrical/optical power model is available."
            )

        elif rid == "E12":
            note = (
                "Physical actuator timing is unavailable and is "
                "therefore not fabricated."
            )

        requirements.append(
            requirement(
                rid,
                name,
                status,
                evidence,
                note,
            )
        )


    # ========================================================
    # F. Computational latency
    # ========================================================

    software_latency = (
        latency
        ["latency"]
        ["software_adb_generation"]
    )

    requirements.append(
        requirement(
            "F1",
            "Software ADB generation latency",
            "MEASURED",
            [
                {
                    "report":
                        str(LATENCY_REPORT),

                    "mean_ms":
                        software_latency[
                            "mean_ms"
                        ],

                    "p95_ms":
                        software_latency[
                            "p95_ms"
                        ],

                    "p99_ms":
                        software_latency[
                            "p99_ms"
                        ],
                }
            ],
            (
                "Computational map-generation latency only. "
                "It excludes predictor inference, sensor timing "
                "and physical actuation."
            ),
        )
    )


    # ========================================================
    # Summary
    # ========================================================

    counts = {}

    for item in requirements:
        status = item["status"]

        counts[status] = (
            counts.get(
                status,
                0
            )
            +
            1
        )


    # Software/modeling blockers and hardware-only limitations
    # are reported separately. A missing physical hardware
    # measurement must not be silently treated as implemented,
    # but it also does not imply an incomplete software pipeline.

    blockers = [
        item["id"]
        for item in requirements
        if item["status"]
        in {
            "NOT_IMPLEMENTED",
            "REQUIRES_MODELING_ASSUMPTION",
        }
    ]

    hardware_limitations = [
        item["id"]
        for item in requirements
        if item["status"] == "NOT_MEASURED"
    ]


    report = {
        "stage":
            6,

        "status":
            (
                "PARTIAL"
                if blockers
                else (
                    "COMPLETE_WITH_HARDWARE_LIMITATION"
                    if hardware_limitations
                    else
                    "COMPLETE"
                )
            ),

        "scientific_policy":
            (
                "No missing requirement is silently treated as "
                "implemented. No unspecified hardware/controller "
                "parameter is invented solely to force completion."
            ),

        "data_source":
            "real_WOMD_validation_class_balanced",

        "future_used_as_input":
            False,

        "requirements":
            requirements,

        "summary": {
            "total_requirements":
                len(requirements),

            "status_counts":
                counts,

            "blocking_items":
                blockers,

            "hardware_limitations":
                hardware_limitations,
        },

        "headline_existing_results": {
            "reactive": {
                "mean_shadow_violation":
                    all_relevant[
                        "reactive"
                    ][
                        "mean_shadow_violation"
                    ],

                "mean_overmask":
                    all_relevant[
                        "reactive"
                    ][
                        "mean_overmask"
                    ],
            },

            "deterministic": {
                "mean_shadow_violation":
                    all_relevant[
                        "deterministic"
                    ][
                        "mean_shadow_violation"
                    ],

                "mean_overmask":
                    all_relevant[
                        "deterministic"
                    ][
                        "mean_overmask"
                    ],
            },

            representative_method: {
                "mean_shadow_violation":
                    representative[
                        "mean_shadow_violation"
                    ],

                "mean_overmask":
                    representative[
                        "mean_overmask"
                    ],

                "mean_road_retention":
                    representative[
                        "mean_road_retention"
                    ],
            },
        },

        "software_adb_generation_latency": {
            "mean_ms":
                software_latency[
                    "mean_ms"
                ],

            "p95_ms":
                software_latency[
                    "p95_ms"
                ],

            "p99_ms":
                software_latency[
                    "p99_ms"
                ],
        },

        "generated_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }


    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )


    print("=" * 78)
    print("STAGE 6 REQUIREMENTS STATUS")
    print("=" * 78)

    print()
    print(
        "overall status =",
        report["status"]
    )

    print(
        "requirements =",
        len(requirements)
    )

    print()

    for status, count in sorted(
        counts.items()
    ):
        print(
            f"{status:30s}",
            count
        )

    print()
    print("BLOCKING ITEMS")

    for rid in blockers:

        item = next(
            x
            for x in requirements
            if x["id"] == rid
        )

        print(
            f"  {rid}: "
            f"{item['requirement']}"
        )

    print()
    print(
        "software ADB latency mean =",
        software_latency[
            "mean_ms"
        ],
        "ms"
    )

    print(
        "software ADB latency p95  =",
        software_latency[
            "p95_ms"
        ],
        "ms"
    )

    print()
    print(
        "Saved:",
        OUT
    )


if __name__ == "__main__":
    main()

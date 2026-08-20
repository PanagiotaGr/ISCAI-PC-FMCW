from pathlib import Path
import json


REPORTS = Path(
    "stage7_real/reports"
)

DATA = Path(
    "stage7_real/data"
)

OUTPUT = REPORTS / "stage7_final_requirements_status.json"


REQUIREMENTS = [
    {
        "id": "S7-1",
        "requirement": "Common posterior",
        "status": "IMPLEMENTED",
        "evidence": [
            str(DATA / "stage7_common_probabilistic_input.npz"),
            str(REPORTS / "stage7_common_probabilistic_input_summary.json"),
        ],
    },
    {
        "id": "S7-2",
        "requirement": "Joint relevant cohort",
        "status": "IMPLEMENTED",
        "evidence": [
            str(DATA / "stage7_joint_cohort.npz"),
            str(REPORTS / "stage7_joint_cohort_summary.json"),
        ],
    },
    {
        "id": "S7-3",
        "requirement": "Primary connected-vehicle communication evaluation",
        "status": "IMPLEMENTED",
        "evidence": [
            str(REPORTS / "stage7_vehicle_communication.json"),
        ],
    },
    {
        "id": "S7-4",
        "requirement": "Joint communication and illumination metrics",
        "status": "IMPLEMENTED",
        "evidence": [
            str(REPORTS / "stage7_final_joint_metrics_pdf_aligned.json"),
        ],
    },
    {
        "id": "S7-5",
        "requirement": "Communication reliability",
        "status": "MEASURED",
        "evidence": [
            str(REPORTS / "stage7_vehicle_communication.json"),
        ],
    },
    {
        "id": "S7-6",
        "requirement": "Beam overhead",
        "status": "MEASURED",
        "evidence": [
            str(REPORTS / "stage7_vehicle_communication.json"),
        ],
    },
    {
        "id": "S7-7",
        "requirement": "Glare protection",
        "status": "MEASURED_WITH_CONSTRUCTED_ORACLE",
        "evidence": [
            str(REPORTS / "stage7_final_joint_metrics_pdf_aligned.json"),
        ],
    },
    {
        "id": "S7-8",
        "requirement": "Pedestrian visibility",
        "status": "MEASURED_WITH_PROXY",
        "evidence": [
            str(REPORTS / "stage7_final_joint_metrics_pdf_aligned.json"),
        ],
    },
    {
        "id": "S7-9",
        "requirement": "Cyclist visibility",
        "status": "MEASURED_WITH_PROXY",
        "evidence": [
            str(REPORTS / "stage7_final_joint_metrics_pdf_aligned.json"),
        ],
    },
    {
        "id": "S7-10",
        "requirement": "Over-masking",
        "status": "MEASURED_WITH_CONSTRUCTED_ORACLE",
        "evidence": [
            str(REPORTS / "stage7_final_joint_metrics_pdf_aligned.json"),
        ],
    },
    {
        "id": "S7-11",
        "requirement": "Uncertainty sweep",
        "status": "IMPLEMENTED",
        "evidence": [
            str(REPORTS / "stage7_uncertainty_sweep.json"),
        ],
    },
    {
        "id": "S7-12",
        "requirement": "Latency sweep",
        "status": "IMPLEMENTED_WITH_MODELING_SCOPE",
        "evidence": [
            str(REPORTS / "stage7_latency_sweep_v2.json"),
        ],
        "note": (
            "Algorithmic stale-posterior latency is evaluated. "
            "Physical ADB actuator latency is not measured."
        ),
    },
    {
        "id": "S7-13",
        "requirement": "Beam codebook sweep",
        "status": "IMPLEMENTED",
        "evidence": [
            str(REPORTS / "stage7_codebook_tradeoff.json"),
        ],
    },
    {
        "id": "S7-14",
        "requirement": "Statistical confidence intervals",
        "status": "IMPLEMENTED",
        "evidence": [
            str(REPORTS / "stage7_joint_confidence_intervals.json"),
            str(DATA / "stage7_joint_actor_metrics.npz"),
        ],
    },
]


def main():

    print("=" * 88)
    print("STAGE 7 FINAL REQUIREMENTS STATUS")
    print("=" * 88)

    missing_paths = []

    for item in REQUIREMENTS:
        for p in item["evidence"]:
            if not Path(p).exists():
                missing_paths.append(
                    (
                        item["id"],
                        p,
                    )
                )

    counts = {}

    for item in REQUIREMENTS:
        status = item["status"]
        counts[status] = (
            counts.get(status, 0)
            + 1
        )

    blockers = []

    if missing_paths:
        blockers.append(
            "MISSING_EVIDENCE_PATHS"
        )

    status = (
        "COMPLETE"
        if not blockers
        else
        "PARTIAL"
    )

    report = {
        "stage": 7,
        "status": status,
        "requirements": REQUIREMENTS,
        "summary": {
            "total_requirements":
                len(REQUIREMENTS),

            "status_counts":
                counts,

            "blocking_items":
                blockers,

            "missing_evidence_paths":
                [
                    {
                        "requirement_id": rid,
                        "path": path,
                    }
                    for rid, path in missing_paths
                ],
        },

        "scientific_policy": (
            "Stage-7 closure requires the shared posterior, "
            "joint communication/illumination evaluation, "
            "uncertainty/latency/codebook sweeps and confidence "
            "intervals. Physical actuator latency is not fabricated "
            "and is explicitly outside the measured Stage-7 scope."
        ),
    }

    OUTPUT.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()
    print("overall status =", status)
    print(
        "requirements =",
        len(REQUIREMENTS)
    )

    print()

    for k in sorted(counts):
        print(
            f"{k:38s}",
            counts[k]
        )

    print()
    print("BLOCKING ITEMS")

    if blockers:
        for x in blockers:
            print(" ", x)

    print()
    print(
        "missing evidence paths =",
        len(missing_paths)
    )

    for rid, path in missing_paths:
        print(
            "MISSING",
            rid,
            path
        )

    print()
    print("Saved:", OUTPUT)

    print()

    if status == "COMPLETE":
        print(
            "STAGE 7 SOFTWARE / MODELING CLOSURE = PASS"
        )
    else:
        print(
            "STAGE 7 SOFTWARE / MODELING CLOSURE = FAIL"
        )


if __name__ == "__main__":
    main()

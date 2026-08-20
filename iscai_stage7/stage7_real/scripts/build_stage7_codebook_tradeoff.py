from pathlib import Path
import json


BEAM = Path(
    "stage7_real/reports/"
    "stage7_joint_beam_evaluation.json"
)

ADB = Path(
    "/home/agni/waymo/iscai_stage6/"
    "stage6_real/reports/"
    "stage6_streaming_evaluation.json"
)

OUTPUT = Path(
    "stage7_real/reports/"
    "stage7_codebook_tradeoff.json"
)

TARGET = 0.95
ADB_POLICY = "class_aware_gamma_0.25"


def main():

    print("=" * 84)
    print("STAGE 7 CODEBOOK / ILLUMINATION TRADE-OFF SWEEP")
    print("=" * 84)

    beam = json.loads(
        BEAM.read_text()
    )

    adb = json.loads(
        ADB.read_text()
    )

    adb_point = (
        adb["results"]
        ["all_relevant"]
        [ADB_POLICY]
    )

    rows = []

    for x in beam["adaptive_topk"]:

        if abs(
            x["target_mass"] - TARGET
        ) > 1e-12:
            continue

        rows.append(
            {
                "num_beams":
                    x["num_beams"],

                "beam_width_deg":
                    x["beam_width_deg"],

                "target_mass":
                    TARGET,

                "communication": {
                    "mean_selected_k":
                        x["selected_k"]["mean"],

                    "p95_selected_k":
                        x["selected_k"]["p95"],

                    "mean_covered_mass":
                        x["covered_mass"]["mean"],

                    "mean_overhead_fraction":
                        x[
                            "mean_overhead_fraction"
                        ],
                },

                "illumination": {
                    "policy":
                        ADB_POLICY,

                    "mean_iou":
                        adb_point["mean_iou"],

                    "mean_shadow_violation":
                        adb_point[
                            "mean_shadow_violation"
                        ],

                    "mean_overmask":
                        adb_point[
                            "mean_overmask"
                        ],

                    "mean_road_retention":
                        adb_point[
                            "mean_road_retention"
                        ],
                },
            }
        )

    rows.sort(
        key=lambda x:
            x["num_beams"]
    )

    report = {
        "status":
            "PASS",

        "stage":
            7,

        "sweep":
            "beam_codebook_size",

        "shared_posterior":
            True,

        "joint_relevant_actors":
            beam["actors"],

        "future_points":
            beam[
                "evaluated_future_points"
            ],

        "fixed_target_mass":
            TARGET,

        "fixed_adb_policy":
            ADB_POLICY,

        "results":
            rows,

        "scientific_scope": (
            "Beam codebook size is varied while "
            "holding posterior target mass and ADB "
            "policy fixed. No scalar joint utility "
            "is imposed."
        ),
    }

    OUTPUT.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()

    for x in rows:

        c = x["communication"]
        i = x["illumination"]

        print(
            f"beams={x['num_beams']:2d} "
            f"width={x['beam_width_deg']:6.2f} deg "
            f"meanK={c['mean_selected_k']:.4f} "
            f"mass={c['mean_covered_mass']:.6f} "
            f"overhead="
            f"{100*c['mean_overhead_fraction']:.3f}% "
            f"IoU={i['mean_iou']:.5f} "
            f"violation="
            f"{100*i['mean_shadow_violation']:.3f}% "
            f"retention="
            f"{i['mean_road_retention']:.5f}"
        )

    print()
    print("Saved:", OUTPUT)
    print()
    print(
        "STAGE 7 CODEBOOK SWEEP = PASS"
    )


if __name__ == "__main__":
    main()

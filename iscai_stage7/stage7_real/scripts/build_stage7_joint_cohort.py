from pathlib import Path
import json
import numpy as np


INPUT = Path(
    "stage7_real/data/"
    "stage7_common_probabilistic_input.npz"
)

OUTPUT = Path(
    "stage7_real/data/"
    "stage7_joint_cohort.npz"
)

REPORT = Path(
    "stage7_real/reports/"
    "stage7_joint_cohort_summary.json"
)


def main():

    print("=" * 82)
    print("STAGE 7 JOINT COMMUNICATION / ILLUMINATION COHORT")
    print("=" * 82)

    d = np.load(INPUT)

    scenario_id = d["scenario_id"]
    track_index = d["track_index"]
    actor_class = d["actor_class"]

    mu = d[
        "trajectory_mean_xy_H_m"
    ]

    current_center = d[
        "current_center_H_m"
    ]

    # ------------------------------------------------------
    # Exact Stage-6 current-FOV definition
    # ------------------------------------------------------

    current_angle = np.degrees(
        np.arctan2(
            current_center[:, 1],
            current_center[:, 0],
        )
    )

    current_range = np.hypot(
        current_center[:, 0],
        current_center[:, 1],
    )

    current_fov = (
        (current_center[:, 0] > 0)
        &
        (np.abs(current_angle) <= 25.0)
        &
        (current_range <= 180.0)
    )

    # ------------------------------------------------------
    # Exact Stage-6 predicted-entry definition
    # ------------------------------------------------------

    pred_angle = np.degrees(
        np.arctan2(
            mu[..., 1],
            mu[..., 0],
        )
    )

    pred_range = np.hypot(
        mu[..., 0],
        mu[..., 1],
    )

    predicted_future_fov = (
        (mu[..., 0] > 0)
        &
        (np.abs(pred_angle) <= 25.0)
        &
        (pred_range <= 180.0)
    )

    predicted_entry = (
        (~current_fov)
        &
        np.any(
            predicted_future_fov,
            axis=1,
        )
    )

    joint_relevant = (
        current_fov
        |
        predicted_entry
    )

    source_index = np.arange(
        len(actor_class),
        dtype=np.int32,
    )

    np.savez_compressed(
        OUTPUT,

        scenario_id=scenario_id,
        track_index=track_index,
        actor_class=actor_class,

        stage6_source_index=source_index,

        current_fov=current_fov,
        predicted_future_fov=predicted_future_fov,
        predicted_entry=predicted_entry,
        joint_relevant=joint_relevant,

        current_angle_deg=current_angle.astype(
            np.float32
        ),
        current_range_m=current_range.astype(
            np.float32
        ),
    )

    classes = [
        "VEHICLE",
        "PEDESTRIAN",
        "CYCLIST",
    ]

    by_class = {}

    for cls in classes:

        c = (
            actor_class.astype(str)
            ==
            cls
        )

        by_class[cls] = {
            "total":
                int(c.sum()),

            "current_fov":
                int(
                    np.logical_and(
                        c,
                        current_fov,
                    ).sum()
                ),

            "predicted_entry":
                int(
                    np.logical_and(
                        c,
                        predicted_entry,
                    ).sum()
                ),

            "joint_relevant":
                int(
                    np.logical_and(
                        c,
                        joint_relevant,
                    ).sum()
                ),
        }

    report = {
        "status":
            "PASS",

        "stage":
            7,

        "source":
            str(INPUT),

        "source_actors":
            int(len(actor_class)),

        "current_fov_actors":
            int(current_fov.sum()),

        "predicted_entry_actors":
            int(predicted_entry.sum()),

        "joint_relevant_actors":
            int(joint_relevant.sum()),

        "future_steps":
            int(mu.shape[1]),

        "fov_definition": {
            "forward":
                "x > 0",

            "absolute_azimuth_deg_max":
                25.0,

            "range_m_max":
                180.0,
        },

        "predicted_entry_definition":
            "not current_fov AND any predicted future mean enters FOV",

        "joint_relevant_definition":
            "current_fov OR predicted_entry",

        "results_by_class":
            by_class,

        "future_used_as_input":
            False,

        "scientific_scope": (
            "Exact reproduction of the Stage-6 causal "
            "evaluation-cohort definition on the shared "
            "Stage-7 probabilistic input."
        ),

        "output":
            str(OUTPUT),
    }

    REPORT.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()
    print("source actors =", len(actor_class))
    print("current-FOV =", int(current_fov.sum()))
    print(
        "predicted-entry =",
        int(predicted_entry.sum()),
    )
    print(
        "joint relevant =",
        int(joint_relevant.sum()),
    )

    print()
    print("===== BY CLASS =====")

    for cls, x in by_class.items():
        print(
            f"{cls:10s} "
            f"total={x['total']:3d} "
            f"current={x['current_fov']:3d} "
            f"entry={x['predicted_entry']:3d} "
            f"joint={x['joint_relevant']:3d}"
        )

    print()
    print("Saved:", OUTPUT)
    print("Report:", REPORT)

    print()
    print("STAGE 7 JOINT COHORT = PASS")


if __name__ == "__main__":
    main()

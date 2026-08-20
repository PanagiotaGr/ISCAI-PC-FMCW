from pathlib import Path
import json
import numpy as np


COMMON = Path(
    "stage7_real/data/"
    "stage7_common_probabilistic_input.npz"
)

COHORT = Path(
    "stage7_real/data/"
    "stage7_joint_cohort.npz"
)

ACTOR_METRICS = Path(
    "stage7_real/data/"
    "stage7_joint_actor_metrics.npz"
)

OUTPUT = Path(
    "stage7_real/data/"
    "stage7_failure_case_slices.npz"
)

REPORT = Path(
    "stage7_real/reports/"
    "stage7_failure_case_slices_summary.json"
)


FOV_MAX_DEG = 25.0
NUM_BEAMS = 64
BEAM_WIDTH_DEG = 360.0 / NUM_BEAMS

# Explicit geometry-based slice thresholds.
BEAM_BOUNDARY_MARGIN_DEG = 0.5
FOV_EDGE_MARGIN_DEG = 3.0
CLOSE_RANGE_M = 20.0
HIGH_ANGULAR_CHANGE_DEG = 5.0


def distance_to_nearest_beam_boundary(angle_deg):
    """
    64-beam codebook spans [-180, 180) uniformly.
    Return minimum angular distance to any beam boundary.
    """
    a = (
        np.asarray(
            angle_deg,
            dtype=np.float64,
        )
        + 180.0
    ) % 360.0

    phase = np.mod(
        a,
        BEAM_WIDTH_DEG,
    )

    return np.minimum(
        phase,
        BEAM_WIDTH_DEG - phase,
    )


def main():

    print("=" * 92)
    print("STAGE 7 FAILURE-CASE SLICE BUILDER")
    print("=" * 92)

    common = np.load(COMMON)
    cohort = np.load(COHORT)
    metrics = np.load(ACTOR_METRICS)

    relevant = cohort[
        "joint_relevant"
    ].astype(bool)

    idx = np.flatnonzero(
        relevant
    )

    if len(idx) != len(
        metrics["actor_class"]
    ):
        raise RuntimeError(
            "Joint cohort / actor-metric size mismatch"
        )

    classes = common[
        "actor_class"
    ][idx].astype(str)

    current_angle = cohort[
        "current_angle_deg"
    ][idx].astype(np.float64)

    current_range = cohort[
        "current_range_m"
    ][idx].astype(np.float64)

    current_fov = cohort[
        "current_fov"
    ][idx].astype(bool)

    predicted_entry = cohort[
        "predicted_entry"
    ][idx].astype(bool)

    mu = common[
        "trajectory_mean_xy_H_m"
    ][idx]

    pred_angle = np.degrees(
        np.arctan2(
            mu[..., 1],
            mu[..., 0],
        )
    )

    # Maximum absolute angular displacement from current bearing.
    delta = (
        pred_angle
        -
        current_angle[:, None]
        +
        180.0
    ) % 360.0 - 180.0

    max_abs_predicted_angle_change = np.max(
        np.abs(delta),
        axis=1,
    )

    beam_boundary_distance = (
        distance_to_nearest_beam_boundary(
            current_angle
        )
    )

    fov_edge_distance = np.abs(
        np.abs(current_angle)
        -
        FOV_MAX_DEG
    )

    near_beam_boundary = (
        beam_boundary_distance
        <=
        BEAM_BOUNDARY_MARGIN_DEG
    )

    near_fov_edge = (
        fov_edge_distance
        <=
        FOV_EDGE_MARGIN_DEG
    )

    close_range = (
        current_range
        <=
        CLOSE_RANGE_M
    )

    high_angular_change = (
        max_abs_predicted_angle_change
        >=
        HIGH_ANGULAR_CHANGE_DEG
    )

    vehicle = (
        classes == "VEHICLE"
    )

    pedestrian = (
        classes == "PEDESTRIAN"
    )

    cyclist = (
        classes == "CYCLIST"
    )

    slices = {
        "near_beam_boundary":
            near_beam_boundary,

        "near_fov_edge":
            near_fov_edge,

        "predicted_entry":
            predicted_entry,

        "close_range":
            close_range,

        "high_predicted_angular_change":
            high_angular_change,

        "vehicle":
            vehicle,

        "pedestrian":
            pedestrian,

        "cyclist":
            cyclist,

        "close_range_and_high_angular_change":
            (
                close_range
                &
                high_angular_change
            ),

        "beam_boundary_and_high_angular_change":
            (
                near_beam_boundary
                &
                high_angular_change
            ),

        "fov_edge_or_predicted_entry":
            (
                near_fov_edge
                |
                predicted_entry
            ),
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT,

        scenario_id=
            common["scenario_id"][idx],

        track_index=
            common["track_index"][idx],

        actor_class=
            classes,

        current_angle_deg=
            current_angle.astype(
                np.float32
            ),

        current_range_m=
            current_range.astype(
                np.float32
            ),

        beam_boundary_distance_deg=
            beam_boundary_distance.astype(
                np.float32
            ),

        fov_edge_distance_deg=
            fov_edge_distance.astype(
                np.float32
            ),

        max_abs_predicted_angle_change_deg=
            max_abs_predicted_angle_change.astype(
                np.float32
            ),

        **{
            name:
                mask.astype(bool)
            for name, mask
            in slices.items()
        },
    )

    summary = {}

    for name, mask in slices.items():

        summary[name] = {
            "actors":
                int(mask.sum()),

            "fraction":
                float(
                    mask.mean()
                ),
        }

    report = {
        "status":
            "PASS",

        "stage":
            7,

        "analysis":
            "failure_case_slice_construction",

        "actors":
            int(len(idx)),

        "definitions": {
            "fov_max_abs_azimuth_deg":
                FOV_MAX_DEG,

            "num_beams":
                NUM_BEAMS,

            "beam_width_deg":
                BEAM_WIDTH_DEG,

            "near_beam_boundary_max_distance_deg":
                BEAM_BOUNDARY_MARGIN_DEG,

            "near_fov_edge_max_distance_deg":
                FOV_EDGE_MARGIN_DEG,

            "close_range_max_m":
                CLOSE_RANGE_M,

            "high_predicted_angular_change_min_deg":
                HIGH_ANGULAR_CHANGE_DEG,
        },

        "slices":
            summary,

        "not_constructed_due_to_missing_direct_fields": [
            "lane_change",
            "braking",
            "intersection_turn",
            "partial_occlusion",
            "temporary_missing_track",
            "low_lidar_point_count",
            "dynamic_blocker",
        ],

        "scientific_scope": (
            "Failure slices are constructed only from "
            "available geometry, class and cohort fields. "
            "No unavailable maneuver, LiDAR visibility, "
            "occlusion or blocker labels are fabricated."
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
    print("actors =", len(idx))
    print(
        "beam width =",
        BEAM_WIDTH_DEG,
        "deg"
    )

    print()
    print("FAILURE / SAFETY SLICES")

    for name, x in summary.items():
        print(
            f"{name:42s} "
            f"actors={x['actors']:3d} "
            f"fraction={100*x['fraction']:.2f}%"
        )

    print()
    print("Saved:", OUTPUT)
    print("Report:", REPORT)

    print()
    print(
        "STAGE 7 FAILURE-CASE SLICES = PASS"
    )


if __name__ == "__main__":
    main()

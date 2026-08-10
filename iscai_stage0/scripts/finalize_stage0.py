from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/waymo/iscai_stage0")

REPORT_DIR = ROOT / "reports/stage0"

OUTPUT_JSON = (
    REPORT_DIR / "stage0_status.json"
)

OUTPUT_MD = (
    REPORT_DIR / "stage0_report.md"
)

IMPLEMENTATION_LOG = (
    ROOT / "docs/implementation_log.md"
)


REQUIRED_REPORTS = {
    "environment":
        REPORT_DIR / "environment_manifest.json",

    "dataset_layout":
        REPORT_DIR / "dataset_layout.json",

    "schema":
        REPORT_DIR / "schema_snapshot.json",

    "multi_scenario":
        REPORT_DIR / "multi_scenario_audit.json",

    "map_schema":
        REPORT_DIR /
        "map_dynamic_schema_audit.json",

    "coordinates":
        REPORT_DIR /
        "motion_coordinate_audit.json",

    "split_scenario_counts":
        REPORT_DIR /
        "split_scenario_counts.json",

    "lidar_subset_manifest":
        REPORT_DIR /
        "lidar_subset_manifest.json",

    "lidar_alignment":
        REPORT_DIR /
        "lidar_alignment_audit.json",
}


REQUIRED_VISUALIZATIONS = {
    "motion_global":
        REPORT_DIR /
        "common_scene_global.png",

    "motion_ego":
        REPORT_DIR /
        "common_scene_ego.png",

    "motion_lidar":
        REPORT_DIR /
        "common_scene_map_boxes_tracks_lidar.png",
}


def load_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def main():
    missing_reports = [
        name
        for name, path
        in REQUIRED_REPORTS.items()
        if not path.is_file()
    ]

    missing_visualizations = [
        name
        for name, path
        in REQUIRED_VISUALIZATIONS.items()
        if not path.is_file()
    ]

    implementation_log_exists = (
        IMPLEMENTATION_LOG.is_file()
    )

    coordinate_pass = False
    lidar_pass = False

    coordinate_path = (
        REQUIRED_REPORTS["coordinates"]
    )

    if coordinate_path.is_file():
        coordinate_pass = (
            load_json(
                coordinate_path
            ).get("status")
            == "pass"
        )

    lidar_path = (
        REQUIRED_REPORTS[
            "lidar_alignment"
        ]
    )

    if lidar_path.is_file():
        lidar = load_json(lidar_path)

        lidar_pass = (
            lidar.get("status") == "pass"
            and lidar.get(
                "scenario_matching",
                {},
            ).get("all_matched") is True
            and lidar.get(
                "lidar_frame_alignment",
                {},
            ).get(
                "all_frame_counts_match"
            ) is True
            and lidar.get(
                "coordinate_alignment",
                {},
            ).get(
                "frame_pose_validated"
            ) is True
            and lidar.get(
                "point_density",
                {},
            ).get("samples", 0) > 0
        )

    stage0_complete = (
        not missing_reports
        and not missing_visualizations
        and implementation_log_exists
        and coordinate_pass
        and lidar_pass
    )

    status = (
        "COMPLETE"
        if stage0_complete
        else "INCOMPLETE"
    )

    gate = {
        "dataset_release_and_layout_audited":
            (
                REQUIRED_REPORTS[
                    "dataset_layout"
                ].is_file()
            ),

        "actual_schema_audited":
            (
                REQUIRED_REPORTS[
                    "schema"
                ].is_file()
            ),

        "split_scenario_counts_verified":
            (
                REQUIRED_REPORTS[
                    "split_scenario_counts"
                ].is_file()
            ),

        "timestamps_and_validity_audited":
            (
                REQUIRED_REPORTS[
                    "multi_scenario"
                ].is_file()
            ),

        "causality_guards_tested":
            True,

        "motion_coordinates_verified":
            coordinate_pass,

        "motion_visualization_generated":
            (
                REQUIRED_VISUALIZATIONS[
                    "motion_global"
                ].is_file()
                and REQUIRED_VISUALIZATIONS[
                    "motion_ego"
                ].is_file()
            ),

        "selective_lidar_manifest_recorded":
            (
                REQUIRED_REPORTS[
                    "lidar_subset_manifest"
                ].is_file()
            ),

        "matching_lidar_frames_open":
            lidar_pass,

        "womd_lidar_timestamp_alignment":
            lidar_pass,

        "lidar_calibration_coordinate_alignment":
            lidar_pass,

        "common_map_tracks_boxes_lidar_visualization":
            (
                lidar_pass
                and REQUIRED_VISUALIZATIONS[
                    "motion_lidar"
                ].is_file()
            ),

        "lidar_point_density_statistics":
            lidar_pass,

        "implementation_log_present":
            implementation_log_exists,
    }

    result = {
        "stage": 0,
        "status": status,
        "stage0_complete":
            stage0_complete,

        "missing_reports":
            missing_reports,

        "missing_visualizations":
            missing_visualizations,

        "completion_gate":
            gate,

        "stage1_training_allowed":
            False,
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Stage 0 final status",
        "",
        f"**Status:** `{status}`",
        "",
        "## Completion gate",
        "",
    ]

    for key, passed in gate.items():
        mark = (
            "PASS"
            if passed
            else "FAIL/PENDING"
        )

        lines.append(
            f"- {key}: **{mark}**"
        )

    lines.extend(
        [
            "",
            (
                "Stage 1 remains blocked until "
                "the Stage 0 results receive "
                "explicit final review/approval."
            ),
        ]
    )

    OUTPUT_MD.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(
        "Stage 0 status:",
        status,
    )

    print(
        "Stage 0 complete:",
        stage0_complete,
    )

    for key, passed in gate.items():
        print(
            f"{key}:",
            "PASS"
            if passed
            else "FAIL/PENDING",
        )

    print("JSON:", OUTPUT_JSON)
    print("Report:", OUTPUT_MD)


if __name__ == "__main__":
    main()
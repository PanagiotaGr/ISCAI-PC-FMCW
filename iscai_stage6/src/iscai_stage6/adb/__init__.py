"""Stage6 illumination-space and ADB primitives."""

from .geometry import (
    Box3D,
    FractionalBoxRegion,
    ProjectedBox,
    ProjectedPoints,
    ProjectedRegion,
    box_corners_headlamp,
    centroid_only_projection,
    project_box_to_headlamp,
    project_points_to_headlamp,
    project_region_to_headlamp,
    region_corners_headlamp,
    validate_controller_provenance,
    wrap_angle,
)

from .grid import (
    IlluminationGridSpec,
    rectangular_extent_cell_indices,
)

from .womd_geometry import (
    CausalADBActorBox,
    SUPPORTED_ADB_CLASSES,
    build_causal_adb_actor_boxes,
    causal_actor_boxes_sha256,
    validate_stage1_projection_consistency,
    world_heading_to_headlamp_yaw,
)

from .illumination import (
    ReactiveShadowRegion,
    angular_shadow_mask,
    combine_illumination_maps_minimum,
    part_a_radial_profile,
    reactive_adb_map,
    single_region_illumination_map,
    static_adb_map,
)

from .part_a_reactive import (
    ORIGINAL_REACTIVE_SUPPORTED_CLASS,
    PART_A_CONTROL_TRANSITION_LENGTH_M,
    PART_A_FIXED_DEMO_EPSILON_DEG,
    PART_A_LATERAL_SAFETY_MARGIN_M,
    PART_A_RADIAL_SAFETY_MARGIN_M,
    PART_A_TARGET_WIDTH_M,
    PART_A_TRZ_TRANSITION_LENGTH_UNITS,
    PartAReactiveGeometry,
    PartAReferenceGrid,
    h0_center_to_part_a_xy,
    part_a_current_vehicle_centers_from_causal_boxes,
    part_a_direct_source_reference_map_from_h0_centers,
    part_a_dynamic_geometry_from_h0_center,
    part_a_original_reactive_map_from_h0_centers,
    part_a_region_from_h0_center,
)

__all__ = (
    "ORIGINAL_REACTIVE_SUPPORTED_CLASS",
    "PART_A_CONTROL_TRANSITION_LENGTH_M",
    "PART_A_FIXED_DEMO_EPSILON_DEG",
    "PART_A_LATERAL_SAFETY_MARGIN_M",
    "PART_A_RADIAL_SAFETY_MARGIN_M",
    "PART_A_TARGET_WIDTH_M",
    "PART_A_TRZ_TRANSITION_LENGTH_UNITS",
    "PartAReactiveGeometry",
    "PartAReferenceGrid",
    "h0_center_to_part_a_xy",
    "part_a_current_vehicle_centers_from_causal_boxes",
    "part_a_direct_source_reference_map_from_h0_centers",
    "part_a_dynamic_geometry_from_h0_center",
    "part_a_original_reactive_map_from_h0_centers",
    "part_a_region_from_h0_center",

    "ReactiveShadowRegion",
    "angular_shadow_mask",
    "combine_illumination_maps_minimum",
    "part_a_radial_profile",
    "reactive_adb_map",
    "single_region_illumination_map",
    "static_adb_map",

    "CausalADBActorBox",
    "SUPPORTED_ADB_CLASSES",
    "build_causal_adb_actor_boxes",
    "causal_actor_boxes_sha256",
    "validate_stage1_projection_consistency",
    "world_heading_to_headlamp_yaw",

    "Box3D",
    "FractionalBoxRegion",
    "ProjectedBox",
    "ProjectedPoints",
    "ProjectedRegion",
    "IlluminationGridSpec",
    "box_corners_headlamp",
    "centroid_only_projection",
    "project_box_to_headlamp",
    "project_points_to_headlamp",
    "project_region_to_headlamp",
    "rectangular_extent_cell_indices",
    "region_corners_headlamp",
    "validate_controller_provenance",
    "wrap_angle",
)

import math
from pathlib import Path

import numpy as np

from iscai_stage0.stage0_geometry import (
    ego_to_global_xy,
    global_to_ego_xy,
)
from iscai_stage0.womd_proto_io import read_first_scenario


TFRECORD = Path(
    "/waymo/data/v1_3_0_scenario/validation/"
    "validation.tfrecord-00000-of-00150"
)


def test_coordinate_roundtrip():
    points = np.array(
        [
            [100.0, 20.0],
            [102.0, 21.0],
            [95.0, 18.0],
        ]
    )

    ego_x = 100.0
    ego_y = 20.0
    heading = 0.37

    local = global_to_ego_xy(
        points,
        ego_x,
        ego_y,
        heading,
    )

    reconstructed = ego_to_global_xy(
        local,
        ego_x,
        ego_y,
        heading,
    )

    assert np.allclose(
        points,
        reconstructed,
        atol=1e-10,
    )


def test_sdc_anchor_maps_to_origin():
    scenario = read_first_scenario(TFRECORD)

    anchor = scenario.current_time_index
    sdc = scenario.tracks[
        scenario.sdc_track_index
    ].states[anchor]

    assert sdc.valid

    local = global_to_ego_xy(
        np.array(
            [[sdc.center_x, sdc.center_y]]
        ),
        sdc.center_x,
        sdc.center_y,
        sdc.heading,
    )

    assert np.allclose(
        local[0],
        [0.0, 0.0],
        atol=1e-10,
    )


def test_ego_forward_direction():
    """
    A point one metre along the SDC heading must map
    approximately to (+1, 0) in the sanity ego frame.
    """
    scenario = read_first_scenario(TFRECORD)

    anchor = scenario.current_time_index
    sdc = scenario.tracks[
        scenario.sdc_track_index
    ].states[anchor]

    point_ahead = np.array(
        [[
            sdc.center_x + math.cos(sdc.heading),
            sdc.center_y + math.sin(sdc.heading),
        ]]
    )

    local = global_to_ego_xy(
        point_ahead,
        sdc.center_x,
        sdc.center_y,
        sdc.heading,
    )

    assert np.allclose(
        local[0],
        [1.0, 0.0],
        atol=1e-9,
    )
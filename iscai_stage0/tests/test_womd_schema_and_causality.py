from pathlib import Path

import pytest

from iscai_stage0.womd_proto_io import (
    CausalScenarioView,
    read_first_scenario,
)


TFRECORD = Path(
    "/waymo/data/v1_3_0_scenario/validation/"
    "validation.tfrecord-00000-of-00150"
)


@pytest.fixture(scope="module")
def scenario():
    return read_first_scenario(TFRECORD)


def test_standard_forecasting_window(scenario):
    assert len(scenario.timestamps_seconds) == 91
    assert scenario.current_time_index == 10

    history_count = scenario.current_time_index
    future_count = (
        len(scenario.timestamps_seconds)
        - scenario.current_time_index
        - 1
    )

    assert history_count == 10
    assert future_count == 80


def test_timestamp_monotonicity_and_rate(scenario):
    timestamps = list(scenario.timestamps_seconds)

    assert all(
        t1 > t0
        for t0, t1 in zip(timestamps[:-1], timestamps[1:])
    )

    deltas = [
        t1 - t0
        for t0, t1 in zip(timestamps[:-1], timestamps[1:])
    ]

    assert all(abs(dt - 0.1) < 1e-3 for dt in deltas)


def test_track_state_alignment(scenario):
    num_timestamps = len(scenario.timestamps_seconds)

    assert len(scenario.tracks) > 0

    for track in scenario.tracks:
        assert len(track.states) == num_timestamps


def test_sdc_index_valid(scenario):
    assert 0 <= scenario.sdc_track_index < len(scenario.tracks)


def test_causal_view_contains_only_past_and_current(scenario):
    view = CausalScenarioView(scenario)

    assert len(view.timestamps) == 11

    for track_index in range(len(scenario.tracks)):
        assert len(view.actor_states(track_index)) == 11


def test_future_state_access_is_blocked(scenario):
    view = CausalScenarioView(scenario)

    with pytest.raises(IndexError, match="Future access forbidden"):
        view.state_at(
            track_index=0,
            time_index=scenario.current_time_index + 1,
        )


def test_current_state_access_is_allowed(scenario):
    view = CausalScenarioView(scenario)

    state = view.state_at(
        track_index=0,
        time_index=scenario.current_time_index,
    )

    assert state is not None


def test_no_velocity_z_in_actual_object_state_schema(scenario):
    state_fields = {
        field.name
        for field in scenario.tracks[0].states[0].DESCRIPTOR.fields
    }

    assert "velocity_x" in state_fields
    assert "velocity_y" in state_fields
    assert "velocity_z" not in state_fields

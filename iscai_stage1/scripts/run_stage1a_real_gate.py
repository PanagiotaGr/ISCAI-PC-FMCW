from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

from iscai_stage0.womd_proto_io import read_first_scenario

from iscai_stage1.contracts.stage1a import HeadlampSurrogateConfig
from iscai_stage1.geometry.frames import (
    SdcStateW,
    build_dynamic_headlamp_frame,
)
from iscai_stage1.io.womd_adapter import (
    build_stage1a_scene_from_womd,
    stage1a_scene_causal_sha256,
)


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "usage: run_stage1a_real_gate.py <motion.tfrecord>",
            file=sys.stderr,
        )
        return 2

    path = Path(sys.argv[1])

    if not path.is_file():
        raise FileNotFoundError(path)

    scenario = read_first_scenario(path)
    scene = build_stage1a_scene_from_womd(scenario)

    anchor = int(scenario.current_time_index)
    assert anchor == scene.anchor_index

    print("scenario_id =", scene.scenario_id)
    print("anchor_index =", anchor)
    print("actor_count =", len(scene.actors))
    print("receiver_count =", len(scene.receivers))

    # --------------------------------------------------------
    # A. Frozen artifact semantics
    # --------------------------------------------------------

    for actor in scene.actors:
        assert (
            actor.semantics.artifact_semantics
            == "causal_womd_annotation_upstream"
        )
        assert actor.semantics.sensor_realistic is False

    print("artifact_semantics_gate = PASS")

    # --------------------------------------------------------
    # B. Every causal actor artifact stops at the anchor
    # --------------------------------------------------------

    for actor in scene.actors:
        history = actor.history

        assert len(history.timestamps_s) == anchor + 1
        assert len(history.position_W_m) == anchor + 1
        assert len(history.state_valid) == anchor + 1
        assert len(history.velocity_W_mps) == anchor + 1
        assert len(history.velocity_valid) == anchor + 1
        assert history.anchor_index == anchor

    print("causal_prefix_gate = PASS")

    # --------------------------------------------------------
    # C. Strict adjacent backward-difference velocity
    # --------------------------------------------------------

    checked_valid_velocity = 0
    checked_invalid_predecessor = 0

    for actor in scene.actors:
        h = actor.history

        for t in range(1, anchor + 1):
            dt = h.timestamps_s[t] - h.timestamps_s[t - 1]

            expected_valid = (
                h.state_valid[t]
                and h.state_valid[t - 1]
                and dt > 0.0
            )

            assert h.velocity_valid[t] == expected_valid

            if expected_valid:
                expected = tuple(
                    (
                        h.position_W_m[t][axis]
                        - h.position_W_m[t - 1][axis]
                    )
                    / dt
                    for axis in range(3)
                )

                for got, want in zip(
                    h.velocity_W_mps[t],
                    expected,
                ):
                    assert math.isclose(
                        got,
                        want,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )

                checked_valid_velocity += 1

            elif (
                h.state_valid[t]
                and not h.state_valid[t - 1]
            ):
                assert h.velocity_W_mps[t] == (
                    0.0,
                    0.0,
                    0.0,
                )
                checked_invalid_predecessor += 1

    print(
        "adjacent_velocity_gate = PASS",
        f"(valid={checked_valid_velocity}, "
        f"invalid_adjacent_predecessor="
        f"{checked_invalid_predecessor})",
    )

    # --------------------------------------------------------
    # D. Forecast candidate does not require anchor velocity
    # --------------------------------------------------------

    for actor in scene.actors:
        causal_valid_count = sum(actor.history.state_valid)

        expected_candidate = (
            actor.history.state_valid[anchor]
            and causal_valid_count >= 2
        )

        assert (
            actor.roles.is_forecasting_target_candidate
            == expected_candidate
        )

    print("forecasting_role_gate = PASS")

    # --------------------------------------------------------
    # E. Ht(anchor) == H0
    # --------------------------------------------------------

    sdc_track = scenario.tracks[scenario.sdc_track_index]
    sdc_anchor = sdc_track.states[anchor]

    state = SdcStateW(
        center_w_m=(
            float(sdc_anchor.center_x),
            float(sdc_anchor.center_y),
            float(sdc_anchor.center_z),
        ),
        heading_rad=float(sdc_anchor.heading),
        length_m=float(sdc_anchor.length),
        valid=bool(sdc_anchor.valid),
    )

    dynamic = build_dynamic_headlamp_frame(
        state,
        HeadlampSurrogateConfig(),
    )

    H0 = scene.anchor_frames.T_W_from_H0
    Ht = dynamic.T_W_from_Ht

    for row in range(3):
        for col in range(3):
            assert math.isclose(
                H0.rotation[row][col],
                Ht.rotation[row][col],
                abs_tol=1e-12,
            )

    for got, want in zip(
        H0.translation,
        Ht.translation,
    ):
        assert math.isclose(
            got,
            want,
            abs_tol=1e-12,
        )

    print("Ht_anchor_equals_H0_gate = PASS")

    # --------------------------------------------------------
    # F. W -> H0 -> W real-data round trip
    # --------------------------------------------------------

    roundtrip_points = 0

    for actor in scene.actors:
        if not actor.history.state_valid[anchor]:
            continue

        point_W = actor.history.position_W_m[anchor]

        point_H0 = (
            scene.anchor_frames.T_H0_from_W.apply_point(
                point_W
            )
        )

        reconstructed_W = (
            scene.anchor_frames.T_W_from_H0.apply_point(
                point_H0
            )
        )

        for got, want in zip(
            reconstructed_W,
            point_W,
        ):
            assert math.isclose(
                got,
                want,
                rel_tol=1e-11,
                abs_tol=1e-8,
            )

        roundtrip_points += 1

    print(
        "real_frame_roundtrip_gate = PASS",
        f"(points={roundtrip_points})",
    )

    # --------------------------------------------------------
    # G. receiver_point_mean_H0 semantics
    # --------------------------------------------------------

    actor_by_track_id = {
        actor.metadata.track_id: actor
        for actor in scene.actors
    }

    for receiver in scene.receivers:
        actor = actor_by_track_id[receiver.track_id]
        geometry = receiver.geometry

        center_W = actor.history.position_W_m[anchor]

        center_H0 = (
            scene.anchor_frames.T_H0_from_W.apply_point(
                center_W
            )
        )

        expected_point = tuple(
            center_H0[i]
            + geometry.receiver_offset_mean_H0[i]
            for i in range(3)
        )

        for got, want in zip(
            geometry.receiver_point_mean_H0,
            expected_point,
        ):
            assert math.isclose(
                got,
                want,
                abs_tol=1e-10,
            )

    print("receiver_point_mean_gate = PASS")

    # --------------------------------------------------------
    # H. Real future-mutation leakage gate
    # --------------------------------------------------------

    original_hash = stage1a_scene_causal_sha256(scene)

    mutated = copy.deepcopy(scenario)

    for track in mutated.tracks:
        for t in range(
            anchor + 1,
            len(track.states),
        ):
            future_state = track.states[t]

            future_state.center_x = (
                float(future_state.center_x)
                + 1_000_000.0
            )
            future_state.center_y = (
                float(future_state.center_y)
                - 1_000_000.0
            )
            future_state.valid = not bool(
                future_state.valid
            )

    for t in range(
        anchor + 1,
        len(mutated.timestamps_seconds),
    ):
        mutated.timestamps_seconds[t] = (
            float(mutated.timestamps_seconds[t])
            + 5000.0
        )

    # Benchmark/evaluation metadata must not influence
    # the Stage-1 causal artifact.
    del mutated.tracks_to_predict[:]
    del mutated.objects_of_interest[:]

    mutated_scene = build_stage1a_scene_from_womd(
        mutated
    )

    mutated_hash = stage1a_scene_causal_sha256(
        mutated_scene
    )

    assert original_hash == mutated_hash

    print(
        "original_causal_sha256 =",
        original_hash,
    )
    print(
        "mutated_causal_sha256  =",
        mutated_hash,
    )
    print("real_future_mutation_gate = PASS")

    print()
    print(
        "STAGE1A_REAL_ACTOR_GEOMETRY_GATE = PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
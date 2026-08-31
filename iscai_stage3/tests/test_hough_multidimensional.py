import math
import unittest

from iscai_stage3.baselines import (
    predict_hough,
)

from iscai_stage3.contracts import (
    AlgorithmObservationSequence,
)

from iscai_stage3.geometry import (
    FrameTransformContext,
)

from iscai_stage3.hough import (
    HoughConfig,
    HoughFrameContext,
    build_raw_hough_frames,
    build_sparse_accumulator,
    extract_hough_peaks,
    run_multidimensional_hough,
)

from rigid_test_factory import (
    T_sensor_from_W,
    identity_transform,
)

from stage2_test_factory import (
    make_detection,
    make_frame,
)


CONFIG = HoughConfig(
    velocity_min_mps=-20.0,
    velocity_max_mps=20.0,
    velocity_step_mps=2.0,
    anchor_position_bin_m=1.0,
    minimum_vote_frames=3,
    minimum_support_frames=3,
    minimum_time_span_s=0.2,
    maximum_peaks=12,
)


def measurement_from_xy(
    *,
    key,
    x,
    y,
    vx,
    vy,
    variance_scale=1.0,
):
    range_m = math.hypot(
        x,
        y,
    )

    az = math.atan2(
        y,
        x,
    )

    radial = (
        (
            x * vx
            +
            y * vy
        )
        /
        range_m
    )

    return make_detection(
        key=key,
        range_m=range_m,
        vr_mps=radial,
        az_rad=az,
        el_rad=0.0,
        variance_scale=(
            variance_scale
        ),
    )


def stationary_context(times):
    return HoughFrameContext(
        transforms=FrameTransformContext(
            T_H0_from_W=(
                identity_transform()
            ),
            T_Ht_from_W_by_frame=tuple(
                identity_transform()
                for _ in times
            ),
        ),
        frame_timestamps_s=tuple(
            times
        ),
    )


def make_linear_sequence(
    *,
    scenario_id,
    times,
    x0,
    y0,
    vx,
    vy,
    include_clutter=False,
    missing_frame=None,
    key_prefix="target",
    variance_scale=1.0,
):
    frames = []

    for index, t in enumerate(times):
        detections = []

        if index != missing_frame:
            x = (
                x0
                +
                vx
                *
                t
            )

            y = (
                y0
                +
                vy
                *
                t
            )

            detections.append(
                measurement_from_xy(
                    key=(
                        f"{key_prefix}-{index}"
                    ),
                    x=x,
                    y=y,
                    vx=vx,
                    vy=vy,
                    variance_scale=(
                        variance_scale
                    ),
                )
            )

        if include_clutter:
            clutter_x = (
                22.0
                -
                3.0
                *
                index
            )

            clutter_y = (
                -8.0
                +
                0.7
                *
                index
                *
                index
            )

            detections.append(
                measurement_from_xy(
                    key=f"clutter-{index}",
                    x=clutter_x,
                    y=clutter_y,
                    vx=0.0,
                    vy=0.0,
                )
            )

        frames.append(
            make_frame(
                timestamp_s=t,
                detections=tuple(
                    detections
                ),
            )
        )

    return AlgorithmObservationSequence(
        scenario_id=scenario_id,
        frames=tuple(frames),
    )


class TestMultidimensionalHough(
    unittest.TestCase
):

    def test_single_linear_track_with_clutter(
        self,
    ):
        times = (
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
        )

        sequence = make_linear_sequence(
            scenario_id="single",
            times=times,
            x0=10.0,
            y0=5.0,
            vx=10.0,
            vy=2.0,
            include_clutter=True,
        )

        result = run_multidimensional_hough(
            sequence,
            context=stationary_context(
                times
            ),
            config=CONFIG,
        )

        self.assertGreaterEqual(
            len(result.tracks),
            1,
        )

        best = min(
            result.tracks,
            key=lambda track: (
                abs(
                    track
                    .velocity_H0_mps[0]
                    -
                    10.0
                )
                +
                abs(
                    track
                    .velocity_H0_mps[1]
                    -
                    2.0
                )
            ),
        )

        self.assertAlmostEqual(
            best.velocity_H0_mps[0],
            10.0,
            delta=1.0,
        )

        self.assertAlmostEqual(
            best.velocity_H0_mps[1],
            2.0,
            delta=1.0,
        )

        self.assertGreaterEqual(
            len(best.support_refs),
            4,
        )

    def test_crossing_tracks(
        self,
    ):
        times = (
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
        )

        frames = []

        for index, t in enumerate(times):
            ax = 10.0 + 4.0 * t
            ay = -2.0 + 10.0 * t

            bx = 10.0 + 4.0 * t
            by = 2.0 - 10.0 * t

            frames.append(
                make_frame(
                    timestamp_s=t,
                    detections=(
                        measurement_from_xy(
                            key=f"a-{index}",
                            x=ax,
                            y=ay,
                            vx=4.0,
                            vy=10.0,
                        ),
                        measurement_from_xy(
                            key=f"b-{index}",
                            x=bx,
                            y=by,
                            vx=4.0,
                            vy=-10.0,
                        ),
                    ),
                )
            )

        result = run_multidimensional_hough(
            AlgorithmObservationSequence(
                scenario_id="crossing",
                frames=tuple(frames),
            ),
            context=stationary_context(
                times
            ),
            config=CONFIG,
        )

        velocities = [
            track.velocity_H0_mps
            for track in result.tracks
        ]

        positive = any(
            abs(v[0] - 4.0) < 1.5
            and
            abs(v[1] - 10.0) < 1.5
            for v in velocities
        )

        negative = any(
            abs(v[0] - 4.0) < 1.5
            and
            abs(v[1] + 10.0) < 1.5
            for v in velocities
        )

        self.assertTrue(
            positive,
            msg=f"velocities={velocities}",
        )

        self.assertTrue(
            negative,
            msg=f"velocities={velocities}",
        )

    def test_missed_detection_tolerated(
        self,
    ):
        times = (
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
        )

        sequence = make_linear_sequence(
            scenario_id="miss",
            times=times,
            x0=10.0,
            y0=0.0,
            vx=10.0,
            vy=0.0,
            missing_frame=2,
        )

        result = run_multidimensional_hough(
            sequence,
            context=stationary_context(
                times
            ),
            config=CONFIG,
        )

        self.assertGreaterEqual(
            len(result.tracks),
            1,
        )

        self.assertTrue(
            any(
                len(track.support_refs)
                >=
                4
                for track in result.tracks
            )
        )

    def test_false_alarm_without_temporal_support_rejected(
        self,
    ):
        times = (
            0.0,
            0.1,
            0.2,
            0.3,
        )

        frames = (
            make_frame(
                timestamp_s=0.0,
                detections=(
                    measurement_from_xy(
                        key="fa0",
                        x=10.0,
                        y=3.0,
                        vx=0.0,
                        vy=0.0,
                    ),
                ),
            ),
            make_frame(
                timestamp_s=0.1,
                detections=(),
            ),
            make_frame(
                timestamp_s=0.2,
                detections=(
                    measurement_from_xy(
                        key="fa2",
                        x=17.0,
                        y=-6.0,
                        vx=0.0,
                        vy=0.0,
                    ),
                ),
            ),
            make_frame(
                timestamp_s=0.3,
                detections=(),
            ),
        )

        result = run_multidimensional_hough(
            AlgorithmObservationSequence(
                scenario_id="false-alarm",
                frames=frames,
            ),
            context=stationary_context(
                times
            ),
            config=CONFIG,
        )

        self.assertEqual(
            result.tracks,
            (),
        )

    def test_measurement_covariance_changes_vote_score(
        self,
    ):
        times = (
            0.0,
            0.1,
            0.2,
        )

        low = make_linear_sequence(
            scenario_id="cov",
            times=times,
            x0=10.0,
            y0=0.0,
            vx=10.0,
            vy=0.0,
            variance_scale=1.0,
        )

        high = make_linear_sequence(
            scenario_id="cov",
            times=times,
            x0=10.0,
            y0=0.0,
            vx=10.0,
            vy=0.0,
            variance_scale=10000.0,
        )

        ctx = stationary_context(
            times
        )

        low_frames = build_raw_hough_frames(
            low,
            context=ctx,
        )

        high_frames = build_raw_hough_frames(
            high,
            context=ctx,
        )

        low_peaks = extract_hough_peaks(
            build_sparse_accumulator(
                low_frames,
                context=ctx,
                config=CONFIG,
            ),
            config=CONFIG,
        )

        high_peaks = extract_hough_peaks(
            build_sparse_accumulator(
                high_frames,
                context=ctx,
                config=CONFIG,
            ),
            config=CONFIG,
        )

        self.assertTrue(
            low_peaks
        )

        self.assertTrue(
            high_peaks
        )

        self.assertGreater(
            low_peaks[0].score,
            high_peaks[0].score,
        )

    def test_detection_keys_do_not_change_result(
        self,
    ):
        times = (
            0.0,
            0.1,
            0.2,
            0.3,
        )

        first = make_linear_sequence(
            scenario_id="keys",
            times=times,
            x0=10.0,
            y0=1.0,
            vx=8.0,
            vy=2.0,
            key_prefix="first",
        )

        second = make_linear_sequence(
            scenario_id="keys",
            times=times,
            x0=10.0,
            y0=1.0,
            vx=8.0,
            vy=2.0,
            key_prefix="totally-different",
        )

        ctx = stationary_context(
            times
        )

        result_a = run_multidimensional_hough(
            first,
            context=ctx,
            config=CONFIG,
        )

        result_b = run_multidimensional_hough(
            second,
            context=ctx,
            config=CONFIG,
        )

        self.assertEqual(
            result_a,
            result_b,
        )

    def test_prediction_closed_form(
        self,
    ):
        times = (
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
        )

        sequence = make_linear_sequence(
            scenario_id="prediction",
            times=times,
            x0=10.0,
            y0=0.0,
            vx=10.0,
            vy=0.0,
        )

        result = run_multidimensional_hough(
            sequence,
            context=stationary_context(
                times
            ),
            config=CONFIG,
        )

        best = min(
            result.tracks,
            key=lambda track: abs(
                track
                .velocity_H0_mps[0]
                -
                10.0
            ),
        )

        prediction = predict_hough(
            best,
            horizons_s=(0.5,),
        )

        expected = (
            best
            .anchor_position_H0_m[0]
            +
            0.5
            *
            best
            .velocity_H0_mps[0]
        )

        self.assertAlmostEqual(
            prediction
            .points[0]
            .position_H0_m[0],
            expected,
            places=10,
        )

    def test_deterministic_repeat(
        self,
    ):
        times = (
            0.0,
            0.1,
            0.2,
            0.3,
        )

        sequence = make_linear_sequence(
            scenario_id="repeat",
            times=times,
            x0=10.0,
            y0=2.0,
            vx=8.0,
            vy=-2.0,
            include_clutter=True,
        )

        ctx = stationary_context(
            times
        )

        first = run_multidimensional_hough(
            sequence,
            context=ctx,
            config=CONFIG,
        )

        second = run_multidimensional_hough(
            sequence,
            context=ctx,
            config=CONFIG,
        )

        self.assertEqual(
            first,
            second,
        )

    def test_raw_cartesian_conversion_compensates_headlamp_motion(
        self,
    ):
        times = (
            0.0,
            0.1,
        )

        sequence = (
            AlgorithmObservationSequence(
                scenario_id="moving-headlamp",
                frames=(
                    make_frame(
                        timestamp_s=0.0,
                        detections=(
                            make_detection(
                                key="k0",
                                range_m=10.0,
                                vr_mps=0.0,
                                az_rad=0.0,
                            ),
                        ),
                    ),
                    make_frame(
                        timestamp_s=0.1,
                        detections=(
                            make_detection(
                                key="k1",
                                range_m=10.0,
                                vr_mps=0.0,
                                az_rad=0.0,
                            ),
                        ),
                    ),
                ),
            )
        )

        ctx = HoughFrameContext(
            transforms=FrameTransformContext(
                T_H0_from_W=(
                    identity_transform()
                ),
                T_Ht_from_W_by_frame=(
                    T_sensor_from_W(
                        sensor_origin_W=(
                            0.0,
                            0.0,
                            0.0,
                        )
                    ),
                    T_sensor_from_W(
                        sensor_origin_W=(
                            1.0,
                            0.0,
                            0.0,
                        )
                    ),
                ),
            ),
            frame_timestamps_s=times,
        )

        frames = build_raw_hough_frames(
            sequence,
            context=ctx,
        )

        self.assertAlmostEqual(
            frames[0]
            .detections[0]
            .position_H0_m[0],
            10.0,
        )

        self.assertAlmostEqual(
            frames[1]
            .detections[0]
            .position_H0_m[0],
            11.0,
        )

    def test_invalid_velocity_grid_rejected(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            HoughConfig(
                velocity_min_mps=10.0,
                velocity_max_mps=-10.0,
            )


if __name__ == "__main__":
    unittest.main()

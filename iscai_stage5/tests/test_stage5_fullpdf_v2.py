from __future__ import annotations

import inspect
import math
import unittest

import numpy as np

from iscai_stage5.fullpdf_v2 import (
    COVERAGE_TARGETS,
    adaptive_topk,
    beam_probability_mass,
    best_link_over_probed_beams,
    build_codebook,
    receiver_angles,
    receiver_geometry,
    receiver_mean_and_covariance_h0,
    receiver_samples_h0,
    recovery_probe_plan,
)


class TestStage5FullPdfV2(
    unittest.TestCase
):

    def test_01_receiver_modes_are_physically_distinct(self):
        centroid = receiver_geometry(
            "centroid",
            length_m=4.0,
            width_m=2.0,
            height_m=1.6,
        )

        known = receiver_geometry(
            "known",
            length_m=4.0,
            width_m=2.0,
            height_m=1.6,
        )

        uncertain = receiver_geometry(
            "uncertain",
            length_m=4.0,
            width_m=2.0,
            height_m=1.6,
        )

        self.assertNotEqual(
            centroid.mean_offset_body_m,
            known.mean_offset_body_m,
        )

        self.assertEqual(
            known.mean_offset_body_m,
            uncertain.mean_offset_body_m,
        )

        self.assertGreater(
            np.trace(
                np.asarray(
                    uncertain
                    .covariance_body_m2
                )
            ),
            0.0,
        )

    def test_02_known_offset_rotates_with_heading(self):
        geometry = receiver_geometry(
            "known",
            length_m=4.0,
            width_m=2.0,
            height_m=1.6,
        )

        mean, covariance = (
            receiver_mean_and_covariance_h0(
                [10.0, 0.0, 0.0],
                np.eye(3),

                heading_h0_rad=(
                    math.pi / 2.0
                ),

                geometry=geometry,
            )
        )

        self.assertAlmostEqual(
            mean[0],
            10.0,
            places=12,
        )

        self.assertAlmostEqual(
            mean[1],
            -2.0,
            places=12,
        )

        self.assertTrue(
            np.allclose(
                covariance,
                np.eye(3),
            )
        )

    def test_03_uncertain_covariance_propagates(self):
        geometry = receiver_geometry(
            "uncertain",
            length_m=4.0,
            width_m=2.0,
            height_m=1.6,
        )

        _, covariance = (
            receiver_mean_and_covariance_h0(
                [10.0, 0.0, 0.0],
                np.eye(3),

                heading_h0_rad=0.0,

                geometry=geometry,
            )
        )

        self.assertGreater(
            covariance[1, 1],
            1.0,
        )

        self.assertGreater(
            covariance[2, 2],
            1.0,
        )

    def test_04_no_calibration_scale_argument_exists(self):
        parameters = (
            inspect.signature(
                receiver_mean_and_covariance_h0
            )
            .parameters
        )

        forbidden = {
            "variance_scale",
            "calibrator",
            "calibration_scale",
        }

        self.assertTrue(
            forbidden.isdisjoint(
                parameters
            )
        )

    def test_05_receiver_mc_is_exactly_reproducible(self):
        geometry = receiver_geometry(
            "uncertain",
            length_m=4.0,
            width_m=2.0,
            height_m=1.6,
        )

        kwargs = dict(
            actor_mean_h0_m=[
                20.0,
                1.0,
                0.5,
            ],

            already_calibrated_predictive_covariance_h0_m2=(
                np.diag(
                    [
                        1.0,
                        0.5,
                        0.2,
                    ]
                )
            ),

            heading_h0_rad=0.2,

            geometry=geometry,

            length_m=4.0,
            width_m=2.0,
            height_m=1.6,

            sample_key="fixed-event",
        )

        a = receiver_samples_h0(
            **kwargs
        )

        b = receiver_samples_h0(
            **kwargs
        )

        self.assertTrue(
            np.array_equal(
                a,
                b,
            )
        )

    def test_06_outside_fov_is_not_clipped(self):
        codebook = build_codebook(
            16
        )

        _, azimuth, _ = (
            receiver_angles(
                [
                    [10.0, 0.0, 0.0],
                ]
            )
        )

        values = np.asarray(
            [
                azimuth[0],
                math.radians(30.0),
            ]
        )

        posterior = (
            beam_probability_mass(
                values,
                codebook,
            )
        )

        self.assertAlmostEqual(
            posterior
            .outside_support_probability,
            0.5,
        )

        self.assertAlmostEqual(
            posterior
            .total_probability,
            1.0,
        )

    def test_07_all_pdf_probability_targets_supported(self):
        self.assertEqual(
            COVERAGE_TARGETS,
            (
                0.90,
                0.95,
                0.975,
                0.99,
            ),
        )

    def test_08_adaptive_topk_is_minimum_mass_set(self):
        codebook = build_codebook(
            16
        )

        # 100 samples:
        # beam 4 = .60
        # beam 5 = .35
        # beam 6 = .05
        samples = []

        for index, count in (
            (4, 60),
            (5, 35),
            (6, 5),
        ):
            samples.extend(
                [
                    codebook[index]
                    .center_azimuth_rad
                ]
                * count
            )

        posterior = (
            beam_probability_mass(
                samples,
                codebook,
            )
        )

        selection = adaptive_topk(
            posterior,
            0.95,
        )

        self.assertEqual(
            selection
            .selected_indices,
            (4, 5),
        )

    def test_09_hysteresis_only_keeps_previous_inside_set(self):
        codebook = build_codebook(
            16
        )

        samples = (
            [
                codebook[4]
                .center_azimuth_rad
            ]
            * 60
            +
            [
                codebook[5]
                .center_azimuth_rad
            ]
            * 40
        )

        posterior = (
            beam_probability_mass(
                samples,
                codebook,
            )
        )

        selection = adaptive_topk(
            posterior,
            0.95,
            previous_primary_index=5,
        )

        self.assertEqual(
            selection.primary_index,
            5,
        )

    def test_10_unattainable_mass_is_explicit_loss_of_lock(self):
        codebook = build_codebook(
            16
        )

        samples = (
            [
                codebook[4]
                .center_azimuth_rad
            ]
            * 70
            +
            [
                math.radians(30.0)
            ]
            * 30
        )

        posterior = (
            beam_probability_mass(
                samples,
                codebook,
            )
        )

        selection = adaptive_topk(
            posterior,
            0.95,
        )

        self.assertFalse(
            selection
            .attainable_inside_support
        )

        self.assertTrue(
            selection
            .loss_of_lock
        )

    def test_11_widened_fallback_is_real_beam_geometry(self):
        codebook = build_codebook(
            64
        )

        samples = (
            [
                codebook[30]
                .center_azimuth_rad
            ]
            * 50
            +
            [
                math.radians(30.0)
            ]
            * 50
        )

        posterior = (
            beam_probability_mass(
                samples,
                codebook,
            )
        )

        selection = adaptive_topk(
            posterior,
            0.95,
        )

        plan = recovery_probe_plan(
            selection,
            codebook,
        )

        fallback = [
            probe.beam
            for probe in plan.probes
            if probe.phase
            == "widened_fallback"
        ]

        self.assertEqual(
            len(fallback),
            1,
        )

        self.assertGreater(
            fallback[0].width_rad,
            codebook[0].width_rad,
        )

    def test_12_no_free_best_link_probes(self):
        codebook = build_codebook(
            16
        )

        samples = [
            codebook[7]
            .center_azimuth_rad
        ] * 100

        posterior = (
            beam_probability_mass(
                samples,
                codebook,
            )
        )

        selection = adaptive_topk(
            posterior,
            0.95,
        )

        plan = recovery_probe_plan(
            selection,
            codebook,
        )

        calls = []

        def evaluator(beam):
            calls.append(
                (
                    beam.codebook_size,
                    beam.index,
                )
            )

            return -abs(
                beam
                .center_azimuth_rad
            )

        result = (
            best_link_over_probed_beams(
                plan,
                evaluator,
            )
        )

        self.assertEqual(
            len(calls),
            plan.probing_beam_count,
        )

        self.assertEqual(
            result[
                "evaluated_beam_count"
            ],
            result[
                "charged_beam_count"
            ],
        )

        self.assertEqual(
            result[
                "free_probe_count"
            ],
            0,
        )

    def test_13_existing_optical_chain_remains_available(self):
        import iscai_stage5.optical_link as optical

        self.assertTrue(
            callable(
                optical.pointing_error
            )
        )

        self.assertTrue(
            callable(
                optical.optical_gain_for_cell
            )
        )

        self.assertTrue(
            callable(
                optical.normalized_received_power
            )
        )

        self.assertTrue(
            callable(
                optical.effective_rate
            )
        )

        self.assertGreater(
            float(
                optical
                .PARTA_REFERENCE_SNR_LINEAR
            ),
            0.0,
        )

    def test_14_existing_latency_contract_uses_probe_count(self):
        import iscai_stage5.beam_latency as latency

        signature = inspect.signature(
            latency.beam_probing_time_s
        )

        self.assertIn(
            "probing_beam_count",
            signature.parameters,
        )


if __name__ == "__main__":
    unittest.main()

import inspect
import unittest

import numpy as np

from iscai_stage5.adaptive_topk_temporal import (
    adaptive_temporal_step,
    initial_adaptive_temporal_state,
)

from iscai_stage5.beam_codebook import (
    BeamProbabilityMass,
)

from iscai_stage5.beam_directional_gain import (
    build_frozen_directional_codebook,
)

from iscai_stage5.beam_latency import (
    BEAM_PROBE_TIMING_PROVENANCE,
    GPU_TIMING_POLICY,
    HIGH_RATE_GROUND_TRUTH_POLICY,
    INFERENCE_DECOMPOSITION,
    PARTA_CHIRP_DURATION_S,
    PDF_LATENCY_FORMULA,
    RATE_FRAME_TIMING_PROVENANCE,
    RUNTIME_CLOCK,
    STAGE5_BEAM_PROBE_DURATION_S,
    STAGE5_RATE_ACCOUNTING_FRAME_S,
    WOMD_SCENE_INTERVAL_S,
    LatencyComponents,
    actual_probe_count_for_adaptive_decision,
    beam_probing_time_s,
    beam_training_overhead_fraction,
    build_latency_budget,
    effective_rate_with_frozen_timing,
    latency_prediction_request,
    measure_call,
)


class Block57Base(
    unittest.TestCase
):

    def probability(
        self,
        beam_count,
        entries,
        *,
        outside=0.0,
    ):
        masses = np.zeros(
            beam_count,
            dtype=np.float64,
        )

        for index, value in entries:
            masses[
                index
            ] = value

        return BeamProbabilityMass(
            masses=tuple(
                float(
                    value
                )
                for value in masses
            ),

            inside_support_mass=float(
                np.sum(
                    masses
                )
            ),

            outside_support_mass=float(
                outside
            ),

            sample_count=2048,
        )


class TestBlock57FrozenTiming(
    Block57Base
):

    def test_01_PartA_chirp_is_10us(self):
        self.assertEqual(
            PARTA_CHIRP_DURATION_S,
            10e-6,
        )

    def test_02_beam_probe_is_one_chirp_baseline(self):
        self.assertEqual(
            STAGE5_BEAM_PROBE_DURATION_S,
            10e-6,
        )

        self.assertIn(
            "constructed_Stage5",
            BEAM_PROBE_TIMING_PROVENANCE,
        )

    def test_03_rate_frame_is_100ms_scene_cycle(self):
        self.assertEqual(
            WOMD_SCENE_INTERVAL_S,
            0.1,
        )

        self.assertEqual(
            STAGE5_RATE_ACCOUNTING_FRAME_S,
            0.1,
        )

        self.assertIn(
            "not_measured_optical",
            RATE_FRAME_TIMING_PROVENANCE,
        )

    def test_04_multi_rate_truth_policy(self):
        self.assertIn(
            "not_annotated_ms_ground_truth",
            HIGH_RATE_GROUND_TRUTH_POLICY,
        )


class TestBlock57ProbingOverhead(
    Block57Base
):

    def test_05_16_beam_exhaustive_overhead(self):
        self.assertAlmostEqual(
            beam_training_overhead_fraction(
                16
            ),
            0.0016,
            places=15,
        )

    def test_06_32_beam_exhaustive_overhead(self):
        self.assertAlmostEqual(
            beam_training_overhead_fraction(
                32
            ),
            0.0032,
            places=15,
        )

    def test_07_64_beam_exhaustive_overhead(self):
        self.assertAlmostEqual(
            beam_training_overhead_fraction(
                64
            ),
            0.0064,
            places=15,
        )

    def test_08_64_beam_zero_BER_effective_rate(self):
        result = (
            effective_rate_with_frozen_timing(
                ber=0.0,
                probing_beam_count=64,
            )
        )

        self.assertAlmostEqual(
            result.effective_rate_bps,
            993_600_000.0,
            places=6,
        )


class TestBlock57LatencyBudget(
    Block57Base
):

    def components(
        self,
        *,
        loading=0.002,
    ):
        return LatencyComponents(
            sensing_s=0.001,
            preprocessing_s=0.002,
            tracking_s=0.003,
            predictor_inference_s=0.004,
            beam_selection_s=0.001,
            actuation_s=0.002,
            data_loading_s=loading,
        )

    def test_09_tau_total_formula(self):
        budget = build_latency_budget(
            components=(
                self.components()
            ),

            probing_beam_count=5,
        )

        expected = (
            0.001
            +
            0.002
            +
            0.003
            +
            0.004
            +
            0.001
            +
            5 * 10e-6
            +
            0.002
        )

        self.assertAlmostEqual(
            budget.online_total_s,
            expected,
            places=15,
        )

        self.assertIn(
            "beam_probing",
            PDF_LATENCY_FORMULA,
        )

    def test_10_inference_decomposed(self):
        budget = build_latency_budget(
            components=(
                self.components()
            ),

            probing_beam_count=1,
        )

        self.assertAlmostEqual(
            budget.inference_total_s,
            0.005,
            places=15,
        )

        self.assertIn(
            "beam_selection",
            INFERENCE_DECOMPOSITION,
        )

    def test_11_data_loading_not_in_online_tau(self):
        first = build_latency_budget(
            components=(
                self.components(
                    loading=0.0
                )
            ),

            probing_beam_count=1,
        )

        second = build_latency_budget(
            components=(
                self.components(
                    loading=10.0
                )
            ),

            probing_beam_count=1,
        )

        self.assertEqual(
            first.online_total_s,
            second.online_total_s,
        )

    def test_12_data_loading_in_wall_runtime(self):
        budget = build_latency_budget(
            components=(
                self.components(
                    loading=0.5
                )
            ),

            probing_beam_count=1,
        )

        self.assertAlmostEqual(
            budget.wall_with_loading_s,
            budget.online_total_s
            +
            0.5,
            places=15,
        )

    def test_13_100ms_deadline(self):
        budget = build_latency_budget(
            components=(
                self.components()
            ),

            probing_beam_count=64,
        )

        self.assertTrue(
            budget.deadline_met
        )

        self.assertEqual(
            budget.deadline_s,
            0.1,
        )


class TestBlock57LatencyPrediction(
    Block57Base
):

    def test_14_sub100ms_target_is_model_based(self):
        components = LatencyComponents(
            sensing_s=0.001,
            preprocessing_s=0.001,
            tracking_s=0.001,
            predictor_inference_s=0.001,
            beam_selection_s=0.001,
            actuation_s=0.001,
        )

        budget = build_latency_budget(
            components=(
                components
            ),

            probing_beam_count=1,
        )

        request = latency_prediction_request(
            budget
        )

        self.assertTrue(
            request.model_based_refinement_required
        )

        self.assertFalse(
            request.annotated_millisecond_ground_truth_used
        )

        self.assertIn(
            "sub_100ms",
            request.target_time_semantics,
        )

    def test_15_exact_100ms_target_uses_frozen_horizon(self):
        components = LatencyComponents(
            sensing_s=0.02,
            preprocessing_s=0.02,
            tracking_s=0.02,
            predictor_inference_s=0.02,
            beam_selection_s=0.01,
            actuation_s=0.01,
        )

        budget = build_latency_budget(
            components=(
                components
            ),

            probing_beam_count=0,
        )

        request = latency_prediction_request(
            budget
        )

        self.assertEqual(
            request.exact_frozen_posterior_horizon_s,
            0.1,
        )

        self.assertFalse(
            request.model_based_refinement_required
        )


class TestBlock57AdaptiveRecoveryAccounting(
    Block57Base
):

    def test_16_normal_adaptive_probe_count_is_K(self):
        codebook = (
            build_frozen_directional_codebook(
                16
            )
        )

        probability = self.probability(
            16,
            (
                (
                    8,
                    0.60,
                ),
                (
                    7,
                    0.35,
                ),
                (
                    9,
                    0.05,
                ),
            ),
        )

        decision, _ = (
            adaptive_temporal_step(
                state=(
                    initial_adaptive_temporal_state()
                ),

                probability=(
                    probability
                ),

                codebook=(
                    codebook
                ),
            )
        )

        self.assertEqual(
            actual_probe_count_for_adaptive_decision(
                decision
            ),
            decision.adaptive_selection.k,
        )

    def test_17_loss_lock_32_beam_is_widened_plus_exhaustive(self):
        codebook = (
            build_frozen_directional_codebook(
                32
            )
        )

        probability = self.probability(
            32,
            (
                (
                    16,
                    0.55,
                ),
                (
                    15,
                    0.30,
                ),
                (
                    17,
                    0.05,
                ),
            ),
            outside=0.10,
        )

        decision, _ = (
            adaptive_temporal_step(
                state=(
                    initial_adaptive_temporal_state()
                ),

                probability=(
                    probability
                ),

                codebook=(
                    codebook
                ),
            )
        )

        self.assertTrue(
            decision.loss_of_lock
        )

        self.assertTrue(
            decision.widened_fallback.available
        )

        self.assertEqual(
            actual_probe_count_for_adaptive_decision(
                decision
            ),
            33,
        )


class TestBlock57InstrumentationAndLeakage(
    Block57Base
):

    def test_18_perf_counter_instrumentation(self):
        result = measure_call(
            lambda:
                123
        )

        self.assertEqual(
            result.value,
            123,
        )

        self.assertGreaterEqual(
            result.elapsed_s,
            0.0,
        )

        self.assertEqual(
            RUNTIME_CLOCK,
            "time.perf_counter_ns",
        )

    def test_19_GPU_sync_policy_explicit(self):
        self.assertIn(
            "synchronize_before_and_after",
            GPU_TIMING_POLICY,
        )

    def test_20_public_latency_APIs_have_no_truth_or_oracle(self):
        APIs = (
            build_latency_budget,
            latency_prediction_request,
            effective_rate_with_frozen_timing,
            actual_probe_count_for_adaptive_decision,
        )

        forbidden = (
            "ground_truth",
            "future_truth",
            "oracle",
            "tracks_to_predict",
            "objects_of_interest",
        )

        for function in APIs:

            signature = str(
                inspect.signature(
                    function
                )
            ).lower()

            for token in forbidden:

                self.assertNotIn(
                    token,
                    signature,
                )


if __name__ == "__main__":
    unittest.main()

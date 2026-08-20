from pathlib import Path
import json
from datetime import datetime, timezone


TOPK_FILE = Path(
    "reports/stage5_real/"
    "real_gaussian_adaptive_topk_with_fixedk.json"
)

LATENCY_FILE = Path(
    "reports/stage5_real/"
    "real_latency_fallback_optical.json"
)

DPSK_FILE = Path(
    "reports/stage5_real/"
    "real_dpsk_snr_sweep.json"
)

OUTPUT = Path(
    "reports/stage5_real/"
    "stage5_final_real_results.json"
)


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r") as f:
        return json.load(f)


def main():

    topk = load_json(TOPK_FILE)
    latency = load_json(LATENCY_FILE)
    dpsk = load_json(DPSK_FILE)

    # --------------------------------------------------------
    # Extract useful summary points
    # --------------------------------------------------------

    adaptive_64_95 = next(
        r
        for r in topk["adaptive_topk"]
        if (
            r["num_beams"] == 64
            and
            abs(r["target_mass"] - 0.95) < 1e-12
        )
    )

    adaptive_64_99 = next(
        r
        for r in topk["adaptive_topk"]
        if (
            r["num_beams"] == 64
            and
            abs(r["target_mass"] - 0.99) < 1e-12
        )
    )

    fixed_64 = [
        r
        for r in topk["fixed_k_baselines"]
        if r["num_beams"] == 64
    ]

    dpsk_summary = dpsk["summary"]

    # --------------------------------------------------------
    # Final package
    # --------------------------------------------------------

    result = {
        "stage": 5,

        "status": "COMPLETE_WITH_SCOPE_NOTE",

        "result_policy": (
            "Real WOMD probabilistic beam-management results "
            "combined with Part-A-grounded SNR-conditioned DPSK "
            "communication evaluation. No fabricated absolute "
            "optical received-power model."
        ),

        "causal_prediction": True,

        "future_used_as_input": False,

        "beam_management": {
            "source": str(TOPK_FILE),

            "evaluation": topk.get("evaluation"),

            "processed_trajectories":
                topk.get("processed_trajectories"),

            "evaluated_future_points":
                topk.get("evaluated_future_points"),

            "codebooks":
                topk.get("codebooks"),

            "targets":
                topk.get("targets"),

            "adaptive_topk":
                topk["adaptive_topk"],

            "fixed_k_baselines":
                topk["fixed_k_baselines"],

            "headline_64_beams": {
                "target_95": adaptive_64_95,
                "target_99": adaptive_64_99,
                "fixed_k": fixed_64,
            },
        },

        "latency_and_fallback": {
            "source": str(LATENCY_FILE),

            "processed_trajectories":
                latency.get("processed_trajectories"),

            "future_point_evaluations":
                latency.get("future_point_evaluations"),

            "latency":
                latency.get("latency"),

            "adaptive_topk":
                latency.get("adaptive_topk"),

            "fallback":
                latency.get("fallback"),
        },

        "communication": {
            "source": str(DPSK_FILE),

            "evaluation":
                dpsk.get("evaluation"),

            "scientific_scope":
                dpsk.get("scientific_scope"),

            "part_a_parameters":
                dpsk.get("part_a_parameters"),

            "snr_conditioned_dpsk":
                dpsk_summary,
        },

        "optical_scope": {
            "absolute_received_power_evaluated": False,

            "hardware_calibrated_link_budget_evaluated": False,

            "reason": (
                "Frozen Part-A artifacts do not document the "
                "complete hardware parameter set required for "
                "absolute P_RX -> SNR evaluation, such as "
                "transmit optical power, receiver responsivity, "
                "aperture/divergence and physical noise-density "
                "parameters."
            ),

            "supported_claim": (
                "Communication performance is evaluated as "
                "DPSK BER/goodput conditioned on SNR using the "
                "Part-A receiver chain."
            ),
        },

        "generated_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    print("=" * 74)
    print("FINAL STAGE 5 REAL RESULTS")
    print("=" * 74)

    print()
    print("64 beams / 95%:")
    print(
        "  mean K =",
        adaptive_64_95["mean_K"],
    )
    print(
        "  p95 K =",
        adaptive_64_95["p95_K"],
    )
    print(
        "  K=1 fraction =",
        adaptive_64_95["fraction_K1"],
    )

    print()
    print("64 beams / 99%:")
    print(
        "  mean K =",
        adaptive_64_99["mean_K"],
    )
    print(
        "  p95 K =",
        adaptive_64_99["p95_K"],
    )

    print()
    print("Latency:")
    print(
        "  inference mean ms/batch =",
        latency["latency"][
            "predictor_inference_mean_ms_per_batch"
        ],
    )
    print(
        "  beam selection mean ms/point =",
        latency["latency"][
            "beam_selection_mean_ms_per_future_point"
        ],
    )

    print()
    print(
        "Fallback counts =",
        latency["fallback"]["counts"],
    )

    print()
    print("DPSK SNR sweep:")
    for row in dpsk_summary:
        print(
            f"  {row['snr_db']:5.1f} dB "
            f"BER={row['aggregate_ber']:.6e} "
            f"goodput="
            f"{row['effective_goodput_bps']/1e9:.6f} Gbps"
        )

    print()
    print(
        "Absolute optical P_RX link budget:",
        "NOT CLAIMED",
    )

    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()

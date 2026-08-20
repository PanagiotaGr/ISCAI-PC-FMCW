from pathlib import Path
import json
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from iscai_stage4.training.womd_dataset import WOMDGRUDataset
from iscai_stage4.training.gru_model import GaussianGRUPredictor

from iscai_stage2.pc_fmcw.reference import FROZEN_PART_A


# ============================================================
# CONFIG
# ============================================================

VAL_DATA = (
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/"
    "validation/motion"
)

CHECKPOINT = Path(
    "/home/agni/waymo/iscai_stage4/"
    "checkpoints/stage4_auto_cuda/"
    "ALL_gaussian_h128_l1_d0.0_lr0.001_b32.pt"
)

OUTPUT = Path(
    "reports/stage5_real/"
    "real_latency_fallback_optical.json"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MAX_FILES = 2
MAX_SAMPLES = 500
BATCH_SIZE = 64

NUM_BEAMS = 64
MC_SAMPLES = 500
TARGET_MASS = 0.95

# fallback policy
MAX_ACTIVE_BEAMS = 5
LOSS_OF_LOCK_THRESHOLD = 0.95


# ============================================================
# GAUSSIAN SAMPLING
# ============================================================

def sample_gaussian(mu, std, rho, n):

    B, T, _ = mu.shape

    z1 = np.random.randn(B, T, n)
    z2 = np.random.randn(B, T, n)

    mux = mu[:, :, 0, None]
    muy = mu[:, :, 1, None]

    sx = std[:, :, 0, None]
    sy = std[:, :, 1, None]

    r = np.clip(
        rho[:, :, None],
        -0.999,
        0.999,
    )

    xs = mux + sx * z1

    ys = (
        muy
        + sy
        * (
            r * z1
            + np.sqrt(1.0 - r*r) * z2
        )
    )

    return xs, ys


# ============================================================
# BEAM POSTERIOR
# ============================================================

def beam_probabilities(
    angles,
    num_beams,
):

    normalized = (
        angles + np.pi
    ) / (
        2.0 * np.pi
    )

    idx = np.floor(
        normalized * num_beams
    ).astype(np.int64)

    idx = np.clip(
        idx,
        0,
        num_beams - 1,
    )

    counts = np.bincount(
        idx,
        minlength=num_beams,
    ).astype(np.float64)

    if counts.sum() > 0:
        counts /= counts.sum()

    return counts


def adaptive_topk(
    probabilities,
    target_mass,
):

    ranked = np.argsort(
        probabilities
    )[::-1]

    selected = []

    mass = 0.0

    for beam in ranked:

        selected.append(
            int(beam)
        )

        mass += float(
            probabilities[beam]
        )

        if mass >= target_mass:
            break

    return selected, mass


# ============================================================
# FALLBACK POLICY
# ============================================================

def fallback_policy(
    selected,
    probabilities,
    previous_beams,
    num_beams,
):

    mode = "adaptive"
    final = list(selected)

    final_mass = float(
        probabilities[final].sum()
    )

    if len(final) > MAX_ACTIVE_BEAMS:

        mode = "widened_fallback"

        final = final[:MAX_ACTIVE_BEAMS]

        final_mass = float(
            probabilities[final].sum()
        )

    if final_mass < LOSS_OF_LOCK_THRESHOLD:

        mode = "loss_of_lock_exhaustive"

        final = list(range(num_beams))
        final_mass = 1.0

    elif previous_beams:

        overlap = [
            b
            for b in previous_beams
            if b in final
        ]

        if overlap:

            mode = "previous_beam_persistence"

            final = (
                overlap
                +
                [
                    b
                    for b in final
                    if b not in overlap
                ]
            )

    return final, mode, final_mass


# ============================================================
# OPTICAL MODEL
# ============================================================

def dpsk_ber_from_snr(
    snr_linear,
):
    """
    Ideal binary DPSK over AWGN:
        BER = 0.5 * exp(-SNR)
    """

    return (
        0.5
        *
        math.exp(
            -snr_linear
        )
    )


def relative_effective_rate(
    beam_count,
    ber,
    total_beams,
    data_rate_bps,
):
    """
    Normalized probing-overhead model.

    No unsupported optical power parameters are invented.

    probing_fraction = K / Ncodebook
    """

    probing_fraction = (
        beam_count
        /
        total_beams
    )

    useful_fraction = max(
        0.0,
        1.0 - probing_fraction,
    )

    return (
        data_rate_bps
        *
        useful_fraction
        *
        (1.0 - ber)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 74)
    print("STAGE 5 REAL LATENCY + FALLBACK + OPTICAL EVALUATION")
    print("=" * 74)

    print("DEVICE:", DEVICE)

    if DEVICE.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print()
    print(
        "Part-A carrier frequency:",
        FROZEN_PART_A.carrier_frequency_hz,
    )

    print(
        "Part-A bandwidth:",
        FROZEN_PART_A.bandwidth_hz,
    )

    print(
        "Part-A data rate:",
        FROZEN_PART_A.data_rate_bps,
    )


    # --------------------------------------------------------
    # DATA LOADING LATENCY
    # --------------------------------------------------------

    t0 = time.perf_counter()

    dataset = WOMDGRUDataset(
        VAL_DATA,
        max_files=MAX_FILES,
        actor_type=[
            "VEHICLE",
            "PEDESTRIAN",
            "CYCLIST",
        ],
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    data_loading_sec = (
        time.perf_counter()
        -
        t0
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = GaussianGRUPredictor(
        input_size=8,
        hidden=128,
        layers=1,
        dropout=0.0,
        future=10,
    ).to(DEVICE)

    state = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state)
    model.eval()


    inference_times = []
    beam_selection_times = []

    selected_k = []
    covered_masses = []

    fallback_counts = {
        "adaptive": 0,
        "previous_beam_persistence": 0,
        "widened_fallback": 0,
        "loss_of_lock_exhaustive": 0,
    }

    # model-based normalized communication metrics
    effective_rates = []

    processed = 0

    previous_beams = []


    with torch.no_grad():

        for x, y in loader:

            remaining = (
                MAX_SAMPLES
                -
                processed
            )

            if remaining <= 0:
                break

            if x.shape[0] > remaining:
                x = x[:remaining]

            x = x.to(
                DEVICE
            )


            # --------------------------------------------
            # Predictor inference latency
            # --------------------------------------------

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            t1 = time.perf_counter()

            pred = model(x)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            inference_times.append(
                time.perf_counter() - t1
            )


            mu = (
                pred["mu"]
                .cpu()
                .numpy()
            )

            std = (
                pred["std"]
                .cpu()
                .numpy()
            )

            rho = (
                pred["rho"][:, :, 0]
                .cpu()
                .numpy()
            )


            xs, ys = sample_gaussian(
                mu,
                std,
                rho,
                MC_SAMPLES,
            )

            B, T, _ = xs.shape


            for i in range(B):

                for t in range(T):

                    angles = np.arctan2(
                        ys[i, t],
                        xs[i, t],
                    )

                    t2 = time.perf_counter()

                    probs = beam_probabilities(
                        angles,
                        NUM_BEAMS,
                    )

                    selected, mass = (
                        adaptive_topk(
                            probs,
                            TARGET_MASS,
                        )
                    )

                    final_beams, mode, final_mass = (
                        fallback_policy(
                            selected,
                            probs,
                            previous_beams,
                            NUM_BEAMS,
                        )
                    )

                    beam_selection_times.append(
                        time.perf_counter()
                        -
                        t2
                    )

                    fallback_counts[
                        mode
                    ] += 1

                    selected_k.append(
                        len(final_beams)
                    )

                    covered_masses.append(
                        final_mass
                    )


                    # ------------------------------------
                    # Model-based optical consequence
                    #
                    # We do NOT invent Ptx/noise-density.
                    # Use posterior mass as normalized
                    # pointing/link success proxy and
                    # convert to a relative SNR scale.
                    # ------------------------------------

                    relative_snr = max(
                        mass,
                        1e-12,
                    )

                    ber = (
                        dpsk_ber_from_snr(
                            relative_snr
                        )
                    )

                    rate = (
                        relative_effective_rate(
                            len(final_beams),
                            ber,
                            NUM_BEAMS,
                            FROZEN_PART_A.data_rate_bps,
                        )
                    )

                    effective_rates.append(
                        rate
                    )

                    previous_beams = list(
                        final_beams
                    )


            processed += B

            print(
                f"samples="
                f"{processed}/{MAX_SAMPLES}",
                flush=True,
            )

            if processed >= MAX_SAMPLES:
                break


    # --------------------------------------------------------
    # AGGREGATE
    # --------------------------------------------------------

    inference_ms = (
        1000.0
        *
        np.asarray(
            inference_times
        )
    )

    beam_ms = (
        1000.0
        *
        np.asarray(
            beam_selection_times
        )
    )

    ks = np.asarray(
        selected_k
    )

    masses = np.asarray(
        covered_masses
    )

    rates = np.asarray(
        effective_rates
    )


    result = {
        "status":
            "PASS_WITH_OPTICAL_PENDING",

        "data_source":
            "real_WOMD_validation",

        "checkpoint":
            str(CHECKPOINT),

        "future_used_as_input":
            False,

        "processed_trajectories":
            processed,

        "future_point_evaluations":
            int(len(ks)),

        "latency": {
            "dataset_loading_sec":
                float(data_loading_sec),

            "predictor_inference_mean_ms_per_batch":
                float(
                    np.mean(
                        inference_ms
                    )
                ),

            "predictor_inference_p95_ms_per_batch":
                float(
                    np.percentile(
                        inference_ms,
                        95,
                    )
                ),

            "beam_selection_mean_ms_per_future_point":
                float(
                    np.mean(
                        beam_ms
                    )
                ),

            "beam_selection_p95_ms_per_future_point":
                float(
                    np.percentile(
                        beam_ms,
                        95,
                    )
                ),
        },

        "adaptive_topk": {
            "target_mass":
                TARGET_MASS,

            "mean_selected_K":
                float(
                    np.mean(
                        ks
                    )
                ),

            "median_selected_K":
                float(
                    np.median(
                        ks
                    )
                ),

            "mean_covered_mass":
                float(
                    np.mean(
                        masses
                    )
                ),
        },

        "fallback": {
            "counts":
                fallback_counts,

            "policy":
                {
                    "max_active_beams":
                        MAX_ACTIVE_BEAMS,

                    "loss_of_lock_threshold":
                        LOSS_OF_LOCK_THRESHOLD,

                    "previous_beam_persistence":
                        True,
                },
        },

        "part_a_reference": {
            "carrier_frequency_hz":
                FROZEN_PART_A.carrier_frequency_hz,

            "bandwidth_hz":
                FROZEN_PART_A.bandwidth_hz,

            "data_rate_bps":
                FROZEN_PART_A.data_rate_bps,

            "wavelength_m":
                FROZEN_PART_A.wavelength_m,
        },

        "optical_evaluation": {
            "semantics":
                (
                    "NON_FINAL PROXY ONLY. "
                    "Not a physical optical link-budget result. "
                    "Ptx, receiver responsivity and physical "
                    "noise-density model are not currently "
                    "documented."
                ),

            "mean_effective_rate_bps":
                float(
                    np.mean(
                        rates
                    )
                ),

            "mean_effective_rate_fraction_of_1Gbps":
                float(
                    np.mean(
                        rates
                    )
                    /
                    FROZEN_PART_A.data_rate_bps
                ),
        },
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


    print()
    print("=" * 74)
    print("RESULTS")
    print("=" * 74)

    print(
        "processed trajectories =",
        processed,
    )

    print(
        "mean K =",
        result[
            "adaptive_topk"
        ][
            "mean_selected_K"
        ],
    )

    print(
        "mean covered mass =",
        result[
            "adaptive_topk"
        ][
            "mean_covered_mass"
        ],
    )

    print()
    print(
        "inference mean ms/batch =",
        result[
            "latency"
        ][
            "predictor_inference_mean_ms_per_batch"
        ],
    )

    print(
        "inference p95 ms/batch =",
        result[
            "latency"
        ][
            "predictor_inference_p95_ms_per_batch"
        ],
    )

    print(
        "beam selection mean ms/point =",
        result[
            "latency"
        ][
            "beam_selection_mean_ms_per_future_point"
        ],
    )

    print()
    print(
        "fallback counts =",
        fallback_counts,
    )

    print()
    print(
        "mean effective rate =",
        result[
            "optical_evaluation"
        ][
            "mean_effective_rate_bps"
        ],
    )

    print(
        "effective-rate fraction =",
        result[
            "optical_evaluation"
        ][
            "mean_effective_rate_fraction_of_1Gbps"
        ],
    )

    print()
    print(
        "Saved:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()

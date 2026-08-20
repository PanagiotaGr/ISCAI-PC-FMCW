from pathlib import Path
import json

import numpy as np


# ============================================================
# PART-A FROZEN PARAMETERS
# ============================================================

C = 299_792_458.0

FC = 193.4e12
B = 10.0e9
T_CHIRP = 10.0e-6

DATA_RATE = 1.0e9
TS = 1.0 / DATA_RATE

N_FAST = 131_072
M_CHIRPS = 64

FS_FAST = N_FAST / T_CHIRP

COMM_CHIRP_INDEX = 0
FFT_ZEROPAD_FACTOR = 8

SNR_DB_VALUES = [
    0.0,
    5.0,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
]

# Use multiple deterministic noise realizations per SNR.
SEEDS = [
    2025,
    2026,
    2027,
    2028,
    2029,
]

OUTPUT = Path(
    "reports/stage5_real/"
    "real_dpsk_snr_sweep.json"
)

EPS = 1e-12


# ============================================================
# SYNTHETIC PART-A PC-FMCW / DPSK WAVEFORM
# ============================================================

def build_transmit_waveform(seed=12345):

    rng = np.random.default_rng(seed)

    t_fast = (
        np.arange(
            N_FAST,
            dtype=np.float64,
        )
        /
        FS_FAST
    )

    num_symbols_per_chirp = int(
        np.ceil(
            T_CHIRP / TS
        )
    )

    dpsk_bits = rng.integers(
        0,
        2,
        size=(
            M_CHIRPS,
            num_symbols_per_chirp,
        ),
    )

    phase_increments = (
        np.pi
        *
        dpsk_bits
    )

    phase_symbols = np.mod(
        np.cumsum(
            phase_increments,
            axis=1,
        ),
        2.0 * np.pi,
    )

    symbol_index_per_sample = (
        np.searchsorted(
            np.arange(
                num_symbols_per_chirp + 1
            )
            * TS,
            t_fast,
            side="right",
        )
        - 1
    )

    symbol_index_per_sample = np.clip(
        symbol_index_per_sample,
        0,
        num_symbols_per_chirp - 1,
    )

    phi_d = phase_symbols[
        :,
        symbol_index_per_sample,
    ]

    chirp_phase = (
        np.pi
        *
        (B / T_CHIRP)
        *
        t_fast**2
    )

    s_tx = np.exp(
        1j
        *
        (
            chirp_phase[None, :]
            +
            phi_d
        )
    )

    return (
        t_fast,
        phi_d,
        s_tx,
    )


# ============================================================
# PART-A DPSK RECEIVER
# ============================================================

def evaluate_one_snr(
    *,
    t_fast,
    phi_d,
    s_tx,
    snr_db,
    noise_seed,
):

    # --------------------------------------------------------
    # Cell 13 - symbol timing
    # --------------------------------------------------------

    comm_num_symbols = int(
        np.floor(
            T_CHIRP / TS
        )
    )

    comm_symbol_edges = np.searchsorted(
        t_fast,
        np.arange(
            comm_num_symbols + 1
        )
        * TS,
        side="left",
    )

    comm_symbol_start_idx = (
        comm_symbol_edges[:-1]
    )

    comm_symbol_stop_idx = (
        comm_symbol_edges[1:]
    )

    valid = (
        comm_symbol_stop_idx
        >
        comm_symbol_start_idx
    )

    comm_symbol_start_idx = (
        comm_symbol_start_idx[
            valid
        ]
    )

    comm_symbol_stop_idx = (
        comm_symbol_stop_idx[
            valid
        ]
    )

    comm_symbol_lengths = (
        comm_symbol_stop_idx
        -
        comm_symbol_start_idx
    )

    comm_num_symbols = len(
        comm_symbol_start_idx
    )

    comm_symbol_center_idx = (
        (
            comm_symbol_start_idx
            +
            comm_symbol_stop_idx
        )
        // 2
    )

    comm_symbol_center_sample = (
        0.5
        *
        (
            comm_symbol_start_idx
            +
            comm_symbol_stop_idx
            -
            1
        )
    )


    # --------------------------------------------------------
    # Cell 14 - AWGN channel
    # --------------------------------------------------------

    comm_tx_waveform = (
        s_tx[
            COMM_CHIRP_INDEX
        ]
    )

    comm_signal_power = np.mean(
        np.abs(
            comm_tx_waveform
        ) ** 2
    )

    comm_noise_power = (
        comm_signal_power
        /
        (
            10.0
            **
            (
                snr_db
                /
                10.0
            )
        )
    )

    comm_rng = np.random.default_rng(
        noise_seed
    )

    comm_noise = (
        np.sqrt(
            comm_noise_power
            /
            2.0
        )
        *
        (
            comm_rng.standard_normal(
                comm_tx_waveform.shape
            )
            +
            1j
            *
            comm_rng.standard_normal(
                comm_tx_waveform.shape
            )
        )
    )

    comm_rx_waveform = (
        comm_tx_waveform
        +
        comm_noise
    )


    # --------------------------------------------------------
    # Cell 15 - FFT carrier extraction
    # --------------------------------------------------------

    comm_fft_len = (
        1
        <<
        int(
            np.ceil(
                np.log2(
                    FFT_ZEROPAD_FACTOR
                    *
                    comm_symbol_lengths.max()
                )
            )
        )
    )

    comm_carrier_cycles_per_sample = (
        np.zeros(
            comm_num_symbols
        )
    )

    comm_carrier_coeffs = (
        np.zeros(
            comm_num_symbols,
            dtype=complex,
        )
    )

    for k in range(
        comm_num_symbols
    ):

        start = int(
            comm_symbol_start_idx[k]
        )

        stop = int(
            comm_symbol_stop_idx[k]
        )

        samples = (
            comm_rx_waveform[
                start:stop
            ]
        )

        spectrum = np.fft.fft(
            samples,
            n=comm_fft_len,
        )

        mag = np.abs(
            spectrum
        )

        peak = int(
            np.argmax(
                mag
            )
        )

        if (
            0
            <
            peak
            <
            comm_fft_len - 1
        ):

            left = np.log(
                mag[
                    peak - 1
                ]
                +
                EPS
            )

            center = np.log(
                mag[
                    peak
                ]
                +
                EPS
            )

            right = np.log(
                mag[
                    peak + 1
                ]
                +
                EPS
            )

            denom = (
                left
                -
                2.0 * center
                +
                right
            )

            if abs(
                denom
            ) > EPS:

                delta = (
                    0.5
                    *
                    (
                        left
                        -
                        right
                    )
                    /
                    denom
                )

                delta = np.clip(
                    delta,
                    -0.5,
                    0.5,
                )

            else:
                delta = 0.0

        else:
            delta = 0.0

        refined_peak = (
            peak
            +
            delta
        )

        mu = (
            refined_peak
            /
            comm_fft_len
        )

        n = np.arange(
            start,
            stop,
        )

        n_centered = (
            n
            -
            comm_symbol_center_sample[
                k
            ]
        )

        coeff = np.mean(
            samples
            *
            np.exp(
                -1j
                *
                2.0
                *
                np.pi
                *
                mu
                *
                n_centered
            )
        )

        comm_carrier_cycles_per_sample[
            k
        ] = mu

        comm_carrier_coeffs[
            k
        ] = coeff


    # --------------------------------------------------------
    # Cell 16 - differential DPSK detection
    # --------------------------------------------------------

    comm_diff_raw = (
        comm_carrier_coeffs[1:]
        *
        np.conj(
            comm_carrier_coeffs[:-1]
        )
    )

    comm_symbol_dn = np.diff(
        comm_symbol_center_sample
    )

    comm_mu_unwrapped = (
        np.unwrap(
            2.0
            *
            np.pi
            *
            comm_carrier_cycles_per_sample
        )
        /
        (
            2.0
            *
            np.pi
        )
    )

    comm_mu_mid = (
        0.5
        *
        (
            comm_mu_unwrapped[1:]
            +
            comm_mu_unwrapped[:-1]
        )
    )

    comm_carrier_phase_step = (
        2.0
        *
        np.pi
        *
        comm_mu_mid
        *
        comm_symbol_dn
    )

    comm_diff_observations = (
        comm_diff_raw
        *
        np.exp(
            -1j
            *
            comm_carrier_phase_step
        )
    )

    comm_bits_hat = (
        np.real(
            comm_diff_observations
        )
        <
        0.0
    )

    comm_tx_symbol_phasors = (
        np.exp(
            1j
            *
            phi_d[
                COMM_CHIRP_INDEX,
                comm_symbol_center_idx,
            ]
        )
    )

    comm_tx_diff = (
        comm_tx_symbol_phasors[1:]
        *
        np.conj(
            comm_tx_symbol_phasors[:-1]
        )
    )

    comm_bits_reference = (
        np.real(
            comm_tx_diff
        )
        <
        0.0
    )

    bit_errors = int(
        np.count_nonzero(
            comm_bits_hat
            !=
            comm_bits_reference
        )
    )

    decisions = int(
        comm_bits_reference.size
    )

    ber = (
        bit_errors
        /
        decisions
    )

    finite_packet_rate_bps = (
        comm_bits_hat.size
        /
        (
            comm_num_symbols
            *
            TS
        )
    )

    return {
        "snr_db":
            float(
                snr_db
            ),

        "noise_seed":
            int(
                noise_seed
            ),

        "signal_power":
            float(
                comm_signal_power
            ),

        "noise_power":
            float(
                comm_noise_power
            ),

        "valid_symbols":
            int(
                comm_num_symbols
            ),

        "bit_decisions":
            decisions,

        "bit_errors":
            bit_errors,

        "ber":
            float(
                ber
            ),

        "finite_packet_rate_bps":
            float(
                finite_packet_rate_bps
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 74)
    print("STAGE 5 PART-A DPSK AWGN SNR SWEEP")
    print("=" * 74)

    print(
        "carrier frequency =",
        FC,
    )

    print(
        "bandwidth =",
        B,
    )

    print(
        "data rate =",
        DATA_RATE,
    )

    print(
        "symbol duration =",
        TS,
    )

    print()

    (
        t_fast,
        phi_d,
        s_tx,
    ) = build_transmit_waveform()

    all_runs = []
    summaries = []

    for snr_db in SNR_DB_VALUES:

        snr_runs = []

        for seed in SEEDS:

            r = evaluate_one_snr(
                t_fast=t_fast,
                phi_d=phi_d,
                s_tx=s_tx,
                snr_db=snr_db,
                noise_seed=seed,
            )

            all_runs.append(
                r
            )

            snr_runs.append(
                r
            )

        bers = np.asarray(
            [
                r["ber"]
                for r in snr_runs
            ],
            dtype=np.float64,
        )

        errors = int(
            sum(
                r["bit_errors"]
                for r in snr_runs
            )
        )

        decisions = int(
            sum(
                r["bit_decisions"]
                for r in snr_runs
            )
        )

        aggregate_ber = (
            errors
            /
            decisions
        )

        summary = {
            "snr_db":
                float(
                    snr_db
                ),

            "noise_realizations":
                len(
                    snr_runs
                ),

            "aggregate_bit_errors":
                errors,

            "aggregate_bit_decisions":
                decisions,

            "aggregate_ber":
                float(
                    aggregate_ber
                ),

            "mean_run_ber":
                float(
                    np.mean(
                        bers
                    )
                ),

            "min_run_ber":
                float(
                    np.min(
                        bers
                    )
                ),

            "max_run_ber":
                float(
                    np.max(
                        bers
                    )
                ),

            "effective_goodput_bps":
                float(
                    DATA_RATE
                    *
                    (
                        1.0
                        -
                        aggregate_ber
                    )
                ),
        }

        summaries.append(
            summary
        )

        print(
            f"SNR={snr_db:5.1f} dB "
            f"errors={errors:6d}/{decisions:6d} "
            f"BER={aggregate_ber:.6e} "
            f"goodput="
            f"{summary['effective_goodput_bps']/1e9:.6f} Gbps"
        )


    report = {
        "status":
            "PASS",

        "evaluation":
            (
                "Part-A-grounded "
                "DPSK AWGN receiver "
                "SNR-conditioned evaluation"
            ),

        "scientific_scope":
            (
                "BER is obtained from the Part-A notebook "
                "AWGN receiver chain. It is not an absolute "
                "optical link-budget result and does not "
                "derive SNR from Ptx/Prx."
            ),

        "part_a_parameters": {
            "carrier_frequency_hz":
                FC,

            "bandwidth_hz":
                B,

            "chirp_duration_s":
                T_CHIRP,

            "data_rate_bps":
                DATA_RATE,

            "symbol_duration_s":
                TS,

            "n_fast":
                N_FAST,

            "m_chirps":
                M_CHIRPS,

            "fft_zeropad_factor":
                FFT_ZEROPAD_FACTOR,
        },

        "snr_db_values":
            SNR_DB_VALUES,

        "noise_seeds":
            SEEDS,

        "summary":
            summaries,

        "runs":
            all_runs,
    }


    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )


    print()
    print(
        "Saved:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import traceback

from iscai_stage1.contracts.stage1a import (
    HEADLAMP_SURROGATE_MODE,
    HeadlampSurrogateConfig,
)

from iscai_stage5.beam_directional_gain import (
    build_frozen_directional_codebook,
)

from iscai_stage5.optical_link import (
    DBPSK_BER_MODEL,
    EFFECTIVE_RATE_MODEL,
    HEADLAMP_FRAME_SEMANTICS,
    MANDATORY_ADDITIONAL_POINTING_LOSS,
    MANDATORY_ATMOSPHERIC_LOSS,
    MANDATORY_GEOMETRIC_LOSS,
    OPTICAL_GAIN_HPBW_RULE,
    OPTICAL_GAIN_MODEL,
    PARTA_RAW_DATA_RATE_BPS,
    PARTA_REFERENCE_NOISE_POWER_NORM,
    PARTA_REFERENCE_SIGNAL_POWER_NORM,
    PARTA_REFERENCE_SNR_DB,
    PARTA_REFERENCE_SNR_LINEAR,
    POWER_UNIT_SEMANTICS,
    dbpsk_ber_from_snr,
    effective_rate,
    evaluate_optical_link,
)


STAGE5 = Path(
    "/home/agni/waymo/iscai_stage5"
)

AUDIT = (
    STAGE5
    / "artifacts/block56/"
      "exact_parta_optical_headlamp_audit.json"
)

BLOCK55 = (
    STAGE5
    / "reports/"
      "block55_adaptive_topk.json"
)

ADAPTIVE_POLICY = (
    STAGE5
    / "configs/"
      "adaptive_topk_policy.json"
)

CODEBOOK_POLICY = (
    STAGE5
    / "configs/"
      "beam_codebook_policy.json"
)

POLICY = (
    STAGE5
    / "configs/"
      "optical_link_policy.json"
)

REPORT = (
    STAGE5
    / "reports/"
      "block56_optical_link.json"
)


EXPECTED_BLOCK55_IMPLEMENTATION_SHA = (
    "a4af90f7463e5ff93c3f6b316188a56b"
    "f25758bb1216d465020155fd6c0205ce"
)

EXPECTED_BLOCK55_POLICY_SHA = (
    "a520f67fd6f74d7c507852cbc4a14c9c"
    "d1e043604706ae4aed89f2775dcddf8f"
)

EXPECTED_CODEBOOK_POLICY_SHA = (
    "bd94f8609393a7c9fe02762cc4bf38e3"
    "a77a90cc31adef5f2e6b06aece4074d7"
)


def require(
    condition,
    message,
):
    if not bool(
        condition
    ):
        raise RuntimeError(
            message
        )


def file_sha256(
    path: Path,
):
    digest = sha256()

    with path.open(
        "rb"
    ) as stream:

        while True:
            chunk = stream.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def load_json(
    path: Path,
):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def write_json(
    path: Path,
    payload,
):
    temporary = path.with_suffix(
        path.suffix
        +
        ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        +
        "\n",
        encoding="utf-8",
    )

    temporary.replace(
        path
    )


def implementation_fingerprint():
    roots = (
        STAGE5 / "src",
        STAGE5 / "tests",
        STAGE5 / "configs",
        STAGE5 / "scripts",
    )

    files = []

    for root in roots:

        if not root.exists():
            continue

        for path in root.rglob(
            "*"
        ):
            if not path.is_file():
                continue

            if "__pycache__" in path.parts:
                continue

            if path.suffix in (
                ".pyc",
                ".pyo",
            ):
                continue

            files.append(
                path
            )

    files.sort(
        key=lambda path:
            str(
                path.relative_to(
                    STAGE5
                )
            )
    )

    digest = sha256()

    for path in files:

        relative = str(
            path.relative_to(
                STAGE5
            )
        )

        digest.update(
            relative.encode(
                "utf-8"
            )
        )

        digest.update(
            b"\0"
        )

        digest.update(
            path.read_bytes()
        )

        digest.update(
            b"\0"
        )

    return (
        len(
            files
        ),
        digest.hexdigest(),
    )


def main():
    print(
        "============================================================"
    )
    print(
        "STAGE 5 — BLOCK 5.6 PART 2/2"
    )
    print(
        "OPTICAL LINK + DPSK + EFFECTIVE-RATE FREEZE"
    )
    print(
        "============================================================"
    )

    required = (
        AUDIT,
        BLOCK55,
        ADAPTIVE_POLICY,
        CODEBOOK_POLICY,
    )

    missing = [
        str(
            path
        )
        for path in required
        if not path.is_file()
    ]

    require(
        not missing,
        (
            "Missing Block5.6 prerequisite(s): "
            +
            ", ".join(
                missing
            )
        ),
    )

    # ========================================================
    # A. Frozen continuity
    # ========================================================

    print()
    print(
        "===== A. FROZEN CONTINUITY ====="
    )

    audit = load_json(
        AUDIT
    )

    require(
        audit.get(
            "status"
        )
        ==
        "PASS_READ_ONLY_AUDIT",
        (
            "Block5.6 source audit "
            "is not PASS."
        ),
    )

    require(
        audit[
            "effective_rate"
        ][
            "source_defined"
        ]
        is False,
        (
            "Part-A unexpectedly claims "
            "a source-defined effective rate."
        ),
    )

    block55 = load_json(
        BLOCK55
    )

    require(
        block55.get(
            "completion"
        )
        ==
        "COMPLETE_FROZEN",
        (
            "Block5.5 is not COMPLETE_FROZEN."
        ),
    )

    require(
        block55[
            "implementation"
        ][
            "sha256"
        ]
        ==
        EXPECTED_BLOCK55_IMPLEMENTATION_SHA,
        (
            "Historical Block5.5 "
            "implementation SHA changed."
        ),
    )

    require(
        file_sha256(
            ADAPTIVE_POLICY
        )
        ==
        EXPECTED_BLOCK55_POLICY_SHA,
        (
            "Frozen adaptive Top-K "
            "policy SHA changed."
        ),
    )

    require(
        file_sha256(
            CODEBOOK_POLICY
        )
        ==
        EXPECTED_CODEBOOK_POLICY_SHA,
        (
            "Frozen codebook policy "
            "SHA changed."
        ),
    )

    print(
        "Block5.5                   = COMPLETE / FROZEN"
    )

    print(
        "adaptive policy SHA        = PASS"
    )

    print(
        "codebook policy SHA        = PASS"
    )

    print(
        "Part-A source audit        = PASS"
    )

    print(
        "Part-A effective_rate      = NOT DEFINED"
    )

    # ========================================================
    # B. H0/headlamp frame resolution
    # ========================================================

    print()
    print(
        "===== B. H0 / HEADLAMP FRAME RESOLUTION ====="
    )

    config = (
        HeadlampSurrogateConfig()
    )

    translation = tuple(
        float(
            value
        )
        for value in (
            config.translation_in_sdc_m(
                4.0
            )
        )
    )

    require(
        translation
        ==
        (
            2.0,
            0.0,
            0.0,
        ),
        (
            "Stage1 default H0 headlamp "
            "translation is not the documented "
            "front-face midpoint baseline."
        ),
    )

    require(
        HEADLAMP_SURROGATE_MODE
        ==
        "front_face_midpoint_surrogate",
        (
            "Stage1 H0 surrogate mode changed."
        ),
    )

    require(
        config.roll_rad
        ==
        0.0
        and
        config.pitch_rad
        ==
        0.0
        and
        config.yaw_rad
        ==
        0.0,
        (
            "Stage1 baseline H0 rotation "
            "is not identity."
        ),
    )

    print(
        "H0 semantic                 = HEADLAMP SURROGATE FRAME"
    )

    print(
        "baseline translation        = [L_SDC/2, 0, 0]"
    )

    print(
        "baseline rotation           = IDENTITY"
    )

    print(
        "configurable extrinsic      = YES"
    )

    print(
        "second H0->headlamp transform = NO"
    )

    # ========================================================
    # C. Part-A normalized communication anchor
    # ========================================================

    print()
    print(
        "===== C. PART-A COMMUNICATION ANCHOR ====="
    )

    require(
        PARTA_RAW_DATA_RATE_BPS
        ==
        1e9,
        (
            "Part-A raw data rate changed."
        ),
    )

    require(
        PARTA_REFERENCE_SIGNAL_POWER_NORM
        ==
        1.0,
        (
            "Part-A normalized signal "
            "reference changed."
        ),
    )

    require(
        PARTA_REFERENCE_NOISE_POWER_NORM
        ==
        0.01,
        (
            "Part-A normalized noise "
            "reference changed."
        ),
    )

    require(
        PARTA_REFERENCE_SNR_LINEAR
        ==
        100.0,
        (
            "Part-A reference SNR changed."
        ),
    )

    require(
        PARTA_REFERENCE_SNR_DB
        ==
        20.0,
        (
            "Part-A reference SNR dB changed."
        ),
    )

    print(
        "raw data rate               = 1.0 Gbps"
    )

    print(
        "reference signal power      = 1 NORMALIZED"
    )

    print(
        "reference noise power       = 0.01 NORMALIZED"
    )

    print(
        "reference SNR               = 100 = 20 dB"
    )

    print(
        "absolute optical Watts      = NOT CLAIMED"
    )

    # ========================================================
    # D. Pointing/gain/SNR/BER chain
    # ========================================================

    print()
    print(
        "===== D. POINTING -> GAIN -> POWER -> SNR -> BER ====="
    )

    codebook = (
        build_frozen_directional_codebook(
            32
        )
    )

    cell = (
        codebook.cells[
            16
        ]
    )

    aligned = evaluate_optical_link(
        receiver_azimuth_rad=(
            cell.center_azimuth_rad
        ),

        receiver_elevation_rad=0.0,

        cell=(
            cell
        ),

        probing_beam_count=3,

        beam_probe_time_s=1e-5,

        frame_time_s=0.1,
    )

    require(
        abs(
            aligned.optical_gain
            -
            1.0
        )
        <
        1e-15,
        (
            "Aligned optical gain is not 1."
        ),
    )

    require(
        abs(
            aligned
            .received_power_normalized
            -
            1.0
        )
        <
        1e-15,
        (
            "Aligned normalized received "
            "power does not reproduce Part-A "
            "reference power."
        ),
    )

    require(
        abs(
            aligned.snr_linear
            -
            100.0
        )
        <
        1e-12,
        (
            "Aligned SNR does not reproduce "
            "Part-A 20-dB reference."
        ),
    )

    require(
        aligned.dbpsk_ber
        <
        1e-40,
        (
            "20-dB analytic DBPSK BER "
            "cross-check failed."
        ),
    )

    print(
        "Gopt model                  = CONSTRUCTED STAGE5 2-D GAUSSIAN"
    )

    print(
        "HPBW azimuth                = CODEBOOK CELL WIDTH"
    )

    print(
        "HPBW elevation              = SAME BASELINE WIDTH"
    )

    print(
        "Lgeo baseline               = 1"
    )

    print(
        "Latm baseline               = 1"
    )

    print(
        "additional Lpoint baseline  = 1"
    )

    print(
        "aligned normalized PRX       = 1 PASS"
    )

    print(
        "aligned SNR                  = 20 dB PASS"
    )

    print(
        "DPSK BER model              = 0.5 exp(-SNR)"
    )

    print(
        "Part-A measured BER curve   = NO"
    )

    # ========================================================
    # E. Effective-rate formula
    # ========================================================

    print()
    print(
        "===== E. EFFECTIVE-RATE FREEZE ====="
    )

    rate = effective_rate(
        ber=0.0,
        probing_beam_count=5,
        beam_probe_time_s=1e-5,
        frame_time_s=0.1,
    )

    require(
        abs(
            rate.raw_overhead_fraction
            -
            0.0005
        )
        <
        1e-15,
        (
            "Beam-probing overhead "
            "formula failed."
        ),
    )

    require(
        abs(
            rate.effective_rate_bps
            -
            999_500_000.0
        )
        <
        1e-6,
        (
            "Effective-rate formula failed."
        ),
    )

    print(
        "Reff formula                 = Rraw*(1-BER)*max(0,1-K*Tbeam/Tframe)"
    )

    print(
        "raw link rate                = 1 Gbps"
    )

    print(
        "K                            = ACTUAL PROBED BEAM COUNT"
    )

    print(
        "Tbeam numeric freeze         = BLOCK5.7"
    )

    print(
        "Tframe numeric freeze        = BLOCK5.7"
    )

    print(
        "formula frozen before formal = YES"
    )

    # ========================================================
    # F. Policy
    # ========================================================

    policy_payload = {
        "project":
            "Agni",

        "stage":
            5,

        "block":
            "5.6",

        "status":
            "FROZEN",

        "headlamp_frame": {
            "frame":
                "H0",

            "semantics":
                HEADLAMP_FRAME_SEMANTICS,

            "Stage1_mode":
                HEADLAMP_SURROGATE_MODE,

            "baseline_translation":
                "[L_SDC/2, 0, 0]",

            "baseline_rotation":
                "identity",

            "second_extrinsic_applied":
                False,
        },

        "PartA_anchor": {
            "raw_data_rate_bps":
                PARTA_RAW_DATA_RATE_BPS,

            "reference_signal_power_normalized":
                PARTA_REFERENCE_SIGNAL_POWER_NORM,

            "reference_noise_power_normalized":
                PARTA_REFERENCE_NOISE_POWER_NORM,

            "reference_snr_linear":
                PARTA_REFERENCE_SNR_LINEAR,

            "reference_snr_db":
                PARTA_REFERENCE_SNR_DB,

            "power_units":
                POWER_UNIT_SEMANTICS,

            "absolute_Watts_claimed":
                False,
        },

        "optical_gain": {
            "model":
                OPTICAL_GAIN_MODEL,

            "HPBW_rule":
                OPTICAL_GAIN_HPBW_RULE,

            "measured":
                False,

            "PartA_source_defined":
                False,

            "constructed_Stage5_extension":
                True,
        },

        "additional_losses": {
            "Lgeo_mandatory_baseline":
                MANDATORY_GEOMETRIC_LOSS,

            "Latm_mandatory_baseline":
                MANDATORY_ATMOSPHERIC_LOSS,

            "Lpoint_extra_mandatory_baseline":
                MANDATORY_ADDITIONAL_POINTING_LOSS,

            "unsupported_range_atmosphere_model_invented":
                False,
        },

        "DPSK_BER": {
            "model":
                DBPSK_BER_MODEL,

            "formula":
                "0.5*exp(-snr_linear)",

            "PartA_role":
                (
                    "analytic_continuation_of_"
                    "frozen_DPSK_AWGN_receiver"
                ),

            "paper_reported_BER_curve":
                False,
        },

        "effective_rate": {
            "model":
                EFFECTIVE_RATE_MODEL,

            "formula":
                (
                    "Rraw*(1-BER)*"
                    "max(0,1-K*Tbeam/Tframe)"
                ),

            "PartA_source_defined":
                False,

            "Stage5_derived":
                True,

            "formula_frozen_in_Block56":
                True,

            "Tbeam_numeric_value":
                "DEFERRED_TO_BLOCK5.7",

            "Tframe_numeric_value":
                "DEFERRED_TO_BLOCK5.7",
        },

        "scope": {
            "formal_N120_read":
                False,

            "Stage4_inference":
                False,

            "training":
                False,

            "recalibration":
                False,

            "absolute_physical_link_budget_claim":
                False,

            "blockage_model":
                False,

            "latency_values_frozen":
                False,
        },
    }

    write_json(
        POLICY,
        policy_payload,
    )

    policy_sha = file_sha256(
        POLICY
    )

    # ========================================================
    # G. Final report
    # ========================================================

    (
        implementation_files,
        implementation_sha,
    ) = implementation_fingerprint()

    report_payload = {
        "project":
            "Agni",

        "stage":
            5,

        "block":
            "5.6",

        "status":
            "PASS",

        "completion":
            "COMPLETE_FROZEN",

        "source_resolution": {
            "PartA_AWGN_DPSK_reference":
                True,

            "PartA_absolute_mobile_optical_link":
                False,

            "PartA_effective_rate":
                False,

            "Stage5_constructed_optical_extension":
                True,
        },

        "headlamp_frame": {
            "H0_is_transmitter_frame":
                True,

            "second_extrinsic":
                False,

            "default_translation_for_L4m":
                list(
                    translation
                ),
        },

        "link_chain": {
            "pointing_error":
                True,

            "optical_gain":
                True,

            "normalized_received_power":
                True,

            "SNR":
                True,

            "DBPSK_BER":
                True,

            "effective_rate":
                True,
        },

        "reference_crosscheck": {
            "aligned_gain":
                aligned.optical_gain,

            "aligned_received_power_normalized":
                aligned
                .received_power_normalized,

            "aligned_snr_linear":
                aligned.snr_linear,

            "aligned_snr_db":
                aligned.snr_db,

            "aligned_DBPSK_BER":
                aligned.dbpsk_ber,
        },

        "effective_rate": {
            "formula":
                (
                    "Rraw*(1-BER)*"
                    "max(0,1-K*Tbeam/Tframe)"
                ),

            "formula_frozen":
                True,

            "numeric_latency_inputs":
                "BLOCK5.7",
        },

        "scope": {
            "formal_N120_read":
                False,

            "Stage4_inference":
                False,

            "training":
                False,

            "recalibration":
                False,

            "absolute_Watt_link_budget":
                False,

            "latency_values_frozen":
                False,
        },

        "policy": {
            "path":
                str(
                    POLICY
                ),

            "sha256":
                policy_sha,
        },

        "implementation": {
            "file_count":
                implementation_files,

            "sha256":
                implementation_sha,
        },

        "next":
            (
                "Block5.7 temporal controller "
                "latency-awareness + numeric "
                "Tbeam/Tframe + fallback/"
                "reacquisition integration"
            ),
    }

    write_json(
        REPORT,
        report_payload,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.6 PART 2/2 GATE"
    )
    print(
        "============================================================"
    )

    print(
        "H0 transmitter/headlamp frame = RESOLVED"
    )

    print(
        "second headlamp extrinsic      = NO"
    )

    print(
        "Part-A raw rate                = 1 Gbps FROZEN"
    )

    print(
        "Part-A reference SNR           = 20 dB FROZEN"
    )

    print(
        "received power units           = NORMALIZED / NOT WATTS"
    )

    print(
        "pointing -> Gopt               = PASS"
    )

    print(
        "Gopt provenance                = CONSTRUCTED STAGE5"
    )

    print(
        "Gopt -> PRX -> SNR             = PASS"
    )

    print(
        "SNR -> DPSK BER                = PASS"
    )

    print(
        "BER -> effective rate          = PASS"
    )

    print(
        "effective-rate formula         = FROZEN"
    )

    print(
        "Tbeam/Tframe values            = BLOCK5.7"
    )

    print(
        "formal N=120 used              = NO"
    )

    print(
        "Stage4 inference               = NO"
    )

    print(
        "training/recalibration         = NO"
    )

    print(
        "policy SHA256                  =",
        policy_sha,
    )

    print(
        "implementation files           =",
        implementation_files,
    )

    print(
        "implementation SHA256          =",
        implementation_sha,
    )

    print(
        "STATUS = PASS"
    )

    print(
        "Block5.6 = COMPLETE / FROZEN"
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "next = BLOCK 5.7"
    )

    print(
        "terminal remains open = YES"
    )


try:
    main()

except BaseException as exc:

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.6 PART 2/2 = BLOCKED"
    )
    print(
        "============================================================"
    )

    print(
        "exception =",
        type(
            exc
        ).__name__,
        str(
            exc
        ),
    )

    print()
    traceback.print_exc()

    print(
        "formal N=120 used = NO"
    )

    print(
        "Stage4 inference = NO"
    )

    print(
        "training/recalibration = NO"
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no sys.exit().

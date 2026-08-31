from __future__ import annotations

import ast
from hashlib import sha256
import json
from pathlib import Path
import traceback

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
    beam_training_overhead_fraction,
    build_latency_budget,
    effective_rate_with_frozen_timing,
    latency_prediction_request,
)


STAGE5 = Path(
    "/home/agni/waymo/iscai_stage5"
)

ROOT = Path(
    "/home/agni/waymo"
)

TIMING_AUDIT = (
    STAGE5
    / "artifacts/block57/"
      "timing_source_audit.json"
)

BLOCK56 = (
    STAGE5
    / "reports/"
      "block56_optical_link.json"
)

OPTICAL_POLICY = (
    STAGE5
    / "configs/"
      "optical_link_policy.json"
)

PARTA_NOTEBOOK = (
    ROOT
    / "part_a_reference/"
      "ISCAI_pc_fmcw/"
      "notebooks/"
      "ISCAI_PC_FMCW.ipynb"
)

STAGE0_TIMING_TEST = (
    ROOT
    / "iscai_stage0/"
      "tests/"
      "test_womd_schema_and_causality.py"
)

POLICY = (
    STAGE5
    / "configs/"
      "beam_latency_policy.json"
)

REPORT = (
    STAGE5
    / "reports/"
      "block57_latency_controller.json"
)


EXPECTED_BLOCK56_IMPLEMENTATION_SHA = (
    "75647e018841a65760b122d619dd11f9"
    "cd5e74e551ece0f64ea19e6189486df9"
)

EXPECTED_BLOCK56_POLICY_SHA = (
    "2075f0d0f3128af94c5d5ed99f0da4d"
    "ba87e08ace69de542bec4c9dd4454302b"
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


def numeric_ast_value(
    node,
):
    if isinstance(
        node,
        ast.Constant,
    ):

        if isinstance(
            node.value,
            (
                int,
                float,
            ),
        ):
            return float(
                node.value
            )

    if isinstance(
        node,
        ast.UnaryOp,
    ):

        value = numeric_ast_value(
            node.operand
        )

        if isinstance(
            node.op,
            ast.USub,
        ):
            return -value

        if isinstance(
            node.op,
            ast.UAdd,
        ):
            return value

    if isinstance(
        node,
        ast.BinOp,
    ):

        left = numeric_ast_value(
            node.left
        )

        right = numeric_ast_value(
            node.right
        )

        if isinstance(
            node.op,
            ast.Mult,
        ):
            return left * right

        if isinstance(
            node.op,
            ast.Div,
        ):
            return left / right

        if isinstance(
            node.op,
            ast.Add,
        ):
            return left + right

        if isinstance(
            node.op,
            ast.Sub,
        ):
            return left - right

        if isinstance(
            node.op,
            ast.Pow,
        ):
            return left ** right

    raise ValueError(
        "Unsupported numeric AST expression."
    )


def resolve_partA_Tchirp():
    payload = load_json(
        PARTA_NOTEBOOK
    )

    values = []

    locations = []

    for index, cell in enumerate(
        payload.get(
            "cells",
            []
        )
    ):

        if cell.get(
            "cell_type"
        ) != "code":
            continue

        source = cell.get(
            "source",
            []
        )

        if isinstance(
            source,
            list,
        ):
            source = "".join(
                source
            )

        try:
            tree = ast.parse(
                str(
                    source
                )
            )

        except SyntaxError:
            continue

        for node in ast.walk(
            tree
        ):

            if not isinstance(
                node,
                ast.Assign,
            ):
                continue

            names = []

            for target in (
                node.targets
            ):

                if isinstance(
                    target,
                    ast.Name,
                ):
                    names.append(
                        target.id
                    )

            if "T_chirp" not in names:
                continue

            try:
                value = numeric_ast_value(
                    node.value
                )

            except Exception:
                continue

            values.append(
                value
            )

            locations.append({
                "cell":
                    index,

                "value_s":
                    value,
            })

    require(
        values,
        (
            "Could not resolve numeric "
            "Part-A T_chirp assignment."
        ),
    )

    require(
        all(
            abs(
                value
                -
                10e-6
            )
            <
            1e-15
            for value in values
        ),
        (
            "Part-A T_chirp assignments "
            "are not consistently 10 us."
        ),
    )

    return (
        10e-6,
        locations,
    )


def verify_stage0_100ms_contract():
    source = STAGE0_TIMING_TEST.read_text(
        encoding="utf-8"
    )

    compact = "".join(
        source.split()
    )

    accepted_patterns = (
        "abs(dt-0.1)<1e-3",
        "abs(dt-0.1)<=1e-3",
    )

    matched = any(
        pattern
        in
        compact
        for pattern in (
            accepted_patterns
        )
    )

    require(
        matched,
        (
            "Could not verify Stage0 "
            "timestamp-delta 0.1-s test."
        ),
    )

    require(
        "timestamps_seconds"
        in
        source,
        (
            "Stage0 timing test does not "
            "reference Scenario timestamps."
        ),
    )

    return {
        "path":
            str(
                STAGE0_TIMING_TEST
            ),

        "sha256":
            file_sha256(
                STAGE0_TIMING_TEST
            ),

        "verified_interval_s":
            0.1,

        "test_tolerance_s":
            1e-3,
    }


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
        "STAGE 5 — BLOCK 5.7 PART 2/2"
    )
    print(
        "TIMING / LATENCY-AWARE CONTROLLER FREEZE"
    )
    print(
        "============================================================"
    )

    required = (
        TIMING_AUDIT,
        BLOCK56,
        OPTICAL_POLICY,
        PARTA_NOTEBOOK,
        STAGE0_TIMING_TEST,
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
            "Missing Block5.7 prerequisite(s): "
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

    timing_audit = load_json(
        TIMING_AUDIT
    )

    require(
        timing_audit.get(
            "status"
        )
        ==
        "PASS_READ_ONLY_AUDIT",
        (
            "Block5.7 Part1 audit "
            "is not PASS."
        ),
    )

    block56 = load_json(
        BLOCK56
    )

    require(
        block56.get(
            "completion"
        )
        ==
        "COMPLETE_FROZEN",
        (
            "Block5.6 is not COMPLETE_FROZEN."
        ),
    )

    require(
        block56[
            "implementation"
        ][
            "sha256"
        ]
        ==
        EXPECTED_BLOCK56_IMPLEMENTATION_SHA,
        (
            "Historical Block5.6 "
            "implementation SHA changed."
        ),
    )

    require(
        file_sha256(
            OPTICAL_POLICY
        )
        ==
        EXPECTED_BLOCK56_POLICY_SHA,
        (
            "Frozen Block5.6 optical "
            "policy SHA changed."
        ),
    )

    print(
        "Block5.6                 = COMPLETE / FROZEN"
    )

    print(
        "historical implementation = PASS"
    )

    print(
        "optical policy SHA       = PASS"
    )

    print(
        "Part1 timing audit       = PASS"
    )

    # ========================================================
    # B. Exact source timing resolution
    # ========================================================

    print()
    print(
        "===== B. EXACT TIMING SOURCE RESOLUTION ====="
    )

    (
        source_Tchirp,
        Tchirp_locations,
    ) = resolve_partA_Tchirp()

    stage0_timing = (
        verify_stage0_100ms_contract()
    )

    require(
        source_Tchirp
        ==
        PARTA_CHIRP_DURATION_S,
        (
            "Runtime Part-A chirp constant "
            "does not match exact source."
        ),
    )

    require(
        stage0_timing[
            "verified_interval_s"
        ]
        ==
        WOMD_SCENE_INTERVAL_S,
        (
            "Runtime WOMD interval constant "
            "does not match Stage0 contract."
        ),
    )

    print(
        "Part-A Tchirp             = 10 us EXACT"
    )

    print(
        "Part-A assignment count   =",
        len(
            Tchirp_locations
        ),
    )

    print(
        "WOMD timestamp interval   = 100 ms VERIFIED"
    )

    print(
        "Stage0 timing-test SHA    =",
        stage0_timing[
            "sha256"
        ],
    )

    # ========================================================
    # C. Freeze Tbeam / Tframe semantics
    # ========================================================

    print()
    print(
        "===== C. TBEAM / TFRAME FREEZE ====="
    )

    require(
        STAGE5_BEAM_PROBE_DURATION_S
        ==
        10e-6,
        (
            "Stage5 Tbeam changed."
        ),
    )

    require(
        STAGE5_RATE_ACCOUNTING_FRAME_S
        ==
        0.1,
        (
            "Stage5 Tframe changed."
        ),
    )

    print(
        "Tbeam                    = 10 us FROZEN"
    )

    print(
        "Tbeam classification     = CONSTRUCTED ONE-CHIRP PROBE BASELINE"
    )

    print(
        "Tbeam measured training  = NO"
    )

    print(
        "Tframe                   = 100 ms FROZEN"
    )

    print(
        "Tframe classification    = WOMD SCENE/CONTROLLER ACCOUNTING CYCLE"
    )

    print(
        "Tframe measured optical frame = NO"
    )

    # ========================================================
    # D. Overhead / effective-rate consequences
    # ========================================================

    print()
    print(
        "===== D. BEAM-PROBING OVERHEAD ====="
    )

    overhead = {}

    zero_ber_rates = {}

    expected = {
        16:
            0.0016,

        32:
            0.0032,

        64:
            0.0064,
    }

    for beam_count in (
        16,
        32,
        64,
    ):

        fraction = (
            beam_training_overhead_fraction(
                beam_count
            )
        )

        require(
            abs(
                fraction
                -
                expected[
                    beam_count
                ]
            )
            <
            1e-15,
            (
                "Beam-training overhead "
                "cross-check failed."
            ),
        )

        rate = (
            effective_rate_with_frozen_timing(
                ber=0.0,

                probing_beam_count=(
                    beam_count
                ),
            )
        )

        overhead[
            str(
                beam_count
            )
        ] = (
            fraction
        )

        zero_ber_rates[
            str(
                beam_count
            )
        ] = (
            rate
            .effective_rate_bps
        )

        print(
            f"{beam_count:2d} beams:"
            f" overhead={100*fraction:.4f}%"
            f" zero-BER Reff="
            f"{rate.effective_rate_bps/1e6:.3f} Mbps"
        )

    # ========================================================
    # E. Latency formula / exact target
    # ========================================================

    print()
    print(
        "===== E. LATENCY-AWARE CONTROLLER CONTRACT ====="
    )

    components = LatencyComponents(
        sensing_s=0.001,
        preprocessing_s=0.002,
        tracking_s=0.003,
        predictor_inference_s=0.004,
        beam_selection_s=0.001,
        actuation_s=0.002,
        data_loading_s=0.010,
    )

    budget = build_latency_budget(
        components=(
            components
        ),

        probing_beam_count=5,
    )

    request = (
        latency_prediction_request(
            budget
        )
    )

    require(
        request.target_offset_s
        ==
        budget.online_total_s,
        (
            "Latency prediction target "
            "is not exactly tau_total."
        ),
    )

    require(
        request
        .annotated_millisecond_ground_truth_used
        is False,
        (
            "Sub-frame latency target "
            "used annotated millisecond GT."
        ),
    )

    print(
        "tau_total formula         =",
        PDF_LATENCY_FORMULA,
    )

    print(
        "tau_inference split       = PREDICTOR + BEAM SELECTION"
    )

    print(
        "data loading in tau_total = NO / REPORTED SEPARATELY"
    )

    print(
        "prediction target         = t + tau_total EXACT"
    )

    print(
        "sub-100ms state source    = CAUSAL MODEL-BASED REFINEMENT"
    )

    print(
        "annotated ms GT           = NO"
    )

    print(
        "100ms online deadline     = EXPLICIT"
    )

    # ========================================================
    # F. Runtime instrumentation
    # ========================================================

    print()
    print(
        "===== F. RUNTIME INSTRUMENTATION CONTRACT ====="
    )

    print(
        "clock                     =",
        RUNTIME_CLOCK,
    )

    print(
        "GPU timing                =",
        GPU_TIMING_POLICY,
    )

    print(
        "measure separately        = DATA LOAD / PREPROCESS / TRACK / PREDICTOR / BEAM SELECT"
    )

    print(
        "beam probing latency      = K * 10 us"
    )

    print(
        "actuation latency         = EXPLICIT COMPONENT"
    )

    print(
        "full Stage5 runtime       = FORMAL MEASUREMENT IN BLOCK5.8"
    )

    # ========================================================
    # G. Freeze policy
    # ========================================================

    policy_payload = {
        "project":
            "Agni",

        "stage":
            5,

        "block":
            "5.7",

        "status":
            "FROZEN",

        "multi_rate_timing": {
            "PartA_Tchirp_s":
                PARTA_CHIRP_DURATION_S,

            "PartA_Tchirp_source":
                "frozen_PartA_notebook_numeric_assignment",

            "Tbeam_s":
                STAGE5_BEAM_PROBE_DURATION_S,

            "Tbeam_provenance":
                BEAM_PROBE_TIMING_PROVENANCE,

            "Tbeam_measured_beam_training_time":
                False,

            "WOMD_scene_interval_s":
                WOMD_SCENE_INTERVAL_S,

            "Tframe_s":
                STAGE5_RATE_ACCOUNTING_FRAME_S,

            "Tframe_provenance":
                RATE_FRAME_TIMING_PROVENANCE,

            "Tframe_measured_optical_frame":
                False,
        },

        "beam_training_overhead": {
            "formula":
                "K*Tbeam/Tframe",

            "exhaustive_fraction":
                overhead,

            "zero_BER_effective_rate_bps":
                zero_ber_rates,
        },

        "latency": {
            "formula":
                PDF_LATENCY_FORMULA,

            "inference_decomposition":
                INFERENCE_DECOMPOSITION,

            "data_loading":
                "reported_separately_not_inside_tau_total",

            "deadline_s":
                WOMD_SCENE_INTERVAL_S,

            "prediction_target":
                "t_plus_tau_total_exact",

            "sub_100ms_policy":
                HIGH_RATE_GROUND_TRUTH_POLICY,

            "annotated_ms_ground_truth":
                False,
        },

        "runtime_measurement": {
            "clock":
                RUNTIME_CLOCK,

            "GPU_policy":
                GPU_TIMING_POLICY,

            "components":
                [
                    "data_loading",
                    "sensing",
                    "preprocessing",
                    "tracking",
                    "predictor_inference",
                    "beam_selection",
                    "beam_probing",
                    "actuation",
                    "online_total",
                ],

            "formal_runtime_measurement":
                "BLOCK5.8",
        },

        "recovery_probe_accounting": {
            "normal":
                "adaptive_mass_covering_K",

            "loss_of_lock":
                (
                    "one_widened_probe_if_available_"
                    "plus_exhaustive_current_codebook"
                ),

            "diagnostic_failed_adaptive_set_charged":
                False,
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

            "future_truth_controller_input":
                False,

            "annotated_sub100ms_GT":
                False,

            "ADB_started":
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
    # H. Final report
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
            "5.7",

        "status":
            "PASS",

        "completion":
            "COMPLETE_FROZEN",

        "timing_resolution": {
            "Tchirp_s":
                PARTA_CHIRP_DURATION_S,

            "Tbeam_s":
                STAGE5_BEAM_PROBE_DURATION_S,

            "Tbeam_is_constructed_mapping":
                True,

            "Tscene_s":
                WOMD_SCENE_INTERVAL_S,

            "Tframe_s":
                STAGE5_RATE_ACCOUNTING_FRAME_S,

            "Tframe_is_optical_measurement":
                False,

            "PartA_assignment_locations":
                Tchirp_locations,

            "Stage0_timestamp_contract":
                stage0_timing,
        },

        "latency_contract": {
            "prediction_target":
                "t_plus_tau_total",

            "sub100ms_GT":
                False,

            "runtime_instrumentation":
                True,

            "formal_runtime_values":
                "BLOCK5.8",
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

            "ADB_started":
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
                "Block5.8 frozen formal Stage5 "
                "evaluation and PDF acceptance gate"
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
        "BLOCK 5.7 PART 2/2 GATE"
    )
    print(
        "============================================================"
    )

    print(
        "Part1 timing audit          = PASS"
    )

    print(
        "Part-A Tchirp               = 10 us VERIFIED"
    )

    print(
        "Tbeam                       = 10 us FROZEN"
    )

    print(
        "Tbeam provenance            = CONSTRUCTED ONE-CHIRP PROBE"
    )

    print(
        "WOMD scene interval         = 100 ms VERIFIED"
    )

    print(
        "Tframe                      = 100 ms FROZEN"
    )

    print(
        "Tframe optical measurement  = NO"
    )

    print(
        "effective-rate timings      = FULLY FROZEN"
    )

    print(
        "latency target              = t + tau_total"
    )

    print(
        "sub-100ms annotated GT      = NO"
    )

    print(
        "runtime instrumentation     = FROZEN"
    )

    print(
        "formal runtime measurement  = BLOCK5.8"
    )

    print(
        "formal N=120 used           = NO"
    )

    print(
        "Stage4 inference            = NO"
    )

    print(
        "training/recalibration      = NO"
    )

    print(
        "policy SHA256               =",
        policy_sha,
    )

    print(
        "implementation files        =",
        implementation_files,
    )

    print(
        "implementation SHA256       =",
        implementation_sha,
    )

    print(
        "STATUS = PASS"
    )

    print(
        "Block5.7 = COMPLETE / FROZEN"
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "next = BLOCK 5.8"
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
        "BLOCK 5.7 PART 2/2 = BLOCKED"
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

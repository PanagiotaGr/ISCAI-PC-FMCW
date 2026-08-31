from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback


ROOT = Path(
    "/home/agni/waymo"
)

S5 = ROOT / "iscai_stage5"
S6 = ROOT / "iscai_stage6"

POLICY = (
    S6
    / "src/iscai_stage6/adb/"
      "class_aware_policy.py"
)

BINDING = (
    S6
    / "src/iscai_stage6/adb/"
      "class_margin_binding.py"
)

TEST = (
    S6
    / "tests/"
      "test_block67_margin_binding.py"
)

REPORT = (
    S6
    / "reports/"
      "block67_part1_margin_binding_gate.json"
)

FAILURE_REPORT = (
    S6
    / "reports/"
      "block67_part1_margin_binding_failure.json"
)

EXPECTED_POLICY_SHA256 = (
    "b998f468b98c2770c48c84a2c0aaaa2"
    "177c2d84b8fc838e80fef9b9da1ebc14b"
)

MIN_FREE_GIB = 250.0


def sha256_file(
    path: Path,
) -> str:

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


def require(
    condition,
    message: str,
):

    if not bool(
        condition
    ):
        raise RuntimeError(
            message
        )


def run_command(
    command,
):

    process = subprocess.run(
        command,
        cwd=str(S6),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    return (
        process.returncode,
        process.stdout,
    )


def test_count(
    output: str,
):

    matches = re.findall(
        r"Ran\s+(\d+)\s+tests?",
        output,
    )

    if not matches:
        return None

    return int(
        matches[-1]
    )


def concise_failure(
    output: str,
    *,
    lines: int = 80,
):

    parts = output.splitlines()

    return "\n".join(
        parts[
            -lines:
        ]
    )


def protected_artifacts():

    paths = [
        S5
        / "reports/stage5_final_closure.json",

        S5
        / "artifacts/block510/"
          "stage5_to_stage6_handoff.json",

        S5
        / "artifacts/block510/"
          "stage5_final_freeze_manifest.json",

        POLICY,
    ]

    for block in range(
        0,
        7,
    ):

        paths.extend(
            sorted(
                (
                    S6 / "reports"
                ).glob(
                    f"block6{block}*.json"
                )
            )
        )

    unique = []

    seen = set()

    for path in paths:

        if (
            path.is_file()
            and
            str(path) not in seen
        ):
            seen.add(
                str(path)
            )
            unique.append(
                path
            )

    return tuple(
        unique
    )


def write_json(
    path: Path,
    payload,
):

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.7 PART 1/2"
    )
    print(
        "CLASS-MARGIN RUNTIME BINDING GATE"
    )
    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # A. Frozen policy
    # --------------------------------------------------------

    print()
    print(
        "===== A. FROZEN CLASS-AWARE POLICY ====="
    )

    require(
        POLICY.is_file(),
        (
            "Missing frozen "
            "class_aware_policy.py"
        ),
    )

    require(
        BINDING.is_file(),
        (
            "Missing Block6.7 "
            "class_margin_binding.py"
        ),
    )

    require(
        TEST.is_file(),
        (
            "Missing Block6.7 "
            "margin-binding test."
        ),
    )

    policy_sha_before = (
        sha256_file(
            POLICY
        )
    )

    require(
        policy_sha_before
        ==
        EXPECTED_POLICY_SHA256,
        (
            "Frozen class_aware_policy.py "
            "SHA changed."
        ),
    )

    protected = (
        protected_artifacts()
    )

    protected_before = {
        str(path):
            sha256_file(
                path
            )
        for path in protected
    }

    print(
        "class_aware_policy SHA =",
        policy_sha_before,
    )

    print(
        "frozen policy exact     = PASS"
    )

    print(
        "frozen policy modified  = NO"
    )

    # --------------------------------------------------------
    # B. Import runtime binding
    # --------------------------------------------------------

    print()
    print(
        "===== B. RUNTIME BINDING IMPORT ====="
    )

    from iscai_stage6.adb.class_aware_policy import (
        MarginComputation,
        TYPE_CYCLIST,
        TYPE_PEDESTRIAN,
        TYPE_VEHICLE,
        angular_margin_cells,
    )

    from iscai_stage6.adb.class_margin_binding import (
        bind_predictive_class_margin,
    )

    print(
        "runtime binding import = PASS"
    )

    # --------------------------------------------------------
    # C. Direct semantic execution
    # --------------------------------------------------------

    print()
    print(
        "===== C. DIRECT m_extra BINDING EXECUTION ====="
    )

    classes = (
        (
            TYPE_VEHICLE,
            0.25,
        ),
        (
            TYPE_PEDESTRIAN,
            0.50,
        ),
        (
            TYPE_CYCLIST,
            0.75,
        ),
    )

    smoke = []

    strict_difference_seen = (
        False
    )

    for (
        actor_class,
        extra,
    ) in classes:

        generic = 1.0

        total = (
            generic
            +
            extra
        )

        margin = MarginComputation(
            actor_class=actor_class,

            predicted_center_H0_xy_m=(
                20.0,
                0.0,
            ),

            predicted_range_m=20.0,

            lateral_predictive_sigma_m=0.0,

            closing_speed_mps=0.0,

            cyclist_lateral_rate_mps=0.0,

            part_a_generic_lateral_margin_m=(
                generic
            ),

            additional_class_margin_m=(
                extra
            ),

            total_lateral_margin_m=(
                total
            ),
        )

        binding = (
            bind_predictive_class_margin(
                margin,
                theta_step_rad=0.01,
            )
        )

        expected_extra_cells = (
            angular_margin_cells(
                total_margin_m=extra,

                predicted_range_m=20.0,

                theta_step_rad=0.01,
            )
        )

        wrong_total_cells = (
            angular_margin_cells(
                total_margin_m=total,

                predicted_range_m=20.0,

                theta_step_rad=0.01,
            )
        )

        require(
            math.isclose(
                binding.predictive_dilation_margin_m,
                extra,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ),
            (
                "Predictive dilation did not "
                "bind to m_extra."
            ),
        )

        require(
            binding.predictive_dilation_cells
            ==
            expected_extra_cells,
            (
                "Runtime dilation cells are "
                "not derived from m_extra."
            ),
        )

        require(
            binding.generic_margin_reapplied
            is False,
            (
                "Part-A generic margin was "
                "incorrectly reapplied."
            ),
        )

        if (
            expected_extra_cells
            !=
            wrong_total_cells
        ):
            strict_difference_seen = (
                True
            )

        smoke.append({
            "actor_class":
                actor_class,

            "part_a_generic_margin_m":
                generic,

            "additional_class_margin_m":
                extra,

            "total_margin_m":
                total,

            "predictive_dilation_cells":
                expected_extra_cells,

            "double_count_reference_cells":
                wrong_total_cells,
        })

        print(
            f"{actor_class:18s}"
            f" | generic={generic:.3f}"
            f" | m_extra={extra:.3f}"
            f" | predictive_cells="
            f"{expected_extra_cells}"
            f" | total_ref_cells="
            f"{wrong_total_cells}"
        )

    require(
        strict_difference_seen,
        (
            "Synthetic gate did not distinguish "
            "m_extra from total-margin dilation."
        ),
    )

    print()
    print(
        "Part3B dilation input = m_extra ONLY PASS"
    )

    print(
        "Part-A generic margin = PRESERVED ONCE PASS"
    )

    print(
        "double counting       = PROHIBITED PASS"
    )

    # --------------------------------------------------------
    # D. Static binding-source contract
    # --------------------------------------------------------

    print()
    print(
        "===== D. SOURCE CONTRACT ====="
    )

    binding_text = BINDING.read_text(
        encoding="utf-8"
    )

    required_tokens = (
        "additional_class_margin_m",
        "predictive_dilation_margin_m=extra",
        "total_margin_m=extra",
        "generic_margin_reapplied=False",
    )

    missing_tokens = [
        token
        for token in required_tokens
        if token not in binding_text
    ]

    require(
        not missing_tokens,
        (
            "Runtime binding source misses "
            "required semantic tokens: "
            + repr(
                missing_tokens
            )
        ),
    )

    print(
        "binding source semantics = PASS"
    )

    print(
        "total margin actuation   = NO"
    )

    print(
        "total margin use         = DIAGNOSTIC ONLY"
    )

    # --------------------------------------------------------
    # E. Targeted regression
    # --------------------------------------------------------

    print()
    print(
        "===== E. BLOCK6.7 TARGETED TESTS ====="
    )

    targeted_rc, targeted_output = (
        run_command(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-p",
                "test_block67_margin_binding.py",
            ]
        )
    )

    targeted_n = test_count(
        targeted_output
    )

    if targeted_rc != 0:
        print(
            concise_failure(
                targeted_output
            )
        )

    require(
        targeted_rc == 0,
        (
            "Block6.7 targeted "
            "margin-binding tests failed."
        ),
    )

    require(
        targeted_n is not None
        and
        targeted_n >= 5,
        (
            "Unexpected Block6.7 "
            "targeted test count."
        ),
    )

    print(
        "targeted tests =",
        targeted_n,
        "/",
        targeted_n,
        "PASS",
    )

    # --------------------------------------------------------
    # F. Full Stage6 regression
    # --------------------------------------------------------

    print()
    print(
        "===== F. FULL STAGE6 REGRESSION ====="
    )

    full_rc, full_output = (
        run_command(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-p",
                "test_*.py",
            ]
        )
    )

    full_n = test_count(
        full_output
    )

    if full_rc != 0:
        print(
            concise_failure(
                full_output
            )
        )

    require(
        full_rc == 0,
        (
            "Full Stage6 regression failed."
        ),
    )

    require(
        full_n is not None
        and
        full_n > 0,
        (
            "Could not verify Stage6 "
            "regression test count."
        ),
    )

    print(
        "full Stage6 regression =",
        full_n,
        "/",
        full_n,
        "PASS",
    )

    # --------------------------------------------------------
    # G. Immutability readback
    # --------------------------------------------------------

    print()
    print(
        "===== G. IMMUTABILITY READBACK ====="
    )

    policy_sha_after = (
        sha256_file(
            POLICY
        )
    )

    require(
        policy_sha_after
        ==
        policy_sha_before
        ==
        EXPECTED_POLICY_SHA256,
        (
            "Frozen class-aware policy changed "
            "during runtime binding."
        ),
    )

    protected_after = {
        str(path):
            sha256_file(
                path
            )
        for path in protected
    }

    changed = [
        path
        for path in protected_before
        if (
            protected_before[
                path
            ]
            !=
            protected_after[
                path
            ]
        )
    ]

    require(
        not changed,
        (
            "Frozen upstream/Stage6 evidence "
            "changed unexpectedly: "
            + repr(
                changed
            )
        ),
    )

    print(
        "class_aware_policy.py = UNCHANGED"
    )

    print(
        "Blocks6.0–6.6 evidence = UNCHANGED"
    )

    print(
        "Stage5 frozen evidence = UNCHANGED"
    )

    # --------------------------------------------------------
    # H. Storage
    # --------------------------------------------------------

    print()
    print(
        "===== H. STORAGE ====="
    )

    free_gib = (
        shutil.disk_usage(
            ROOT
        ).free
        /
        1024**3
    )

    require(
        free_gib
        >=
        MIN_FREE_GIB,
        (
            "250-GiB reserve violated."
        ),
    )

    print(
        "free GiB        =",
        round(
            free_gib,
            3,
        ),
    )

    print(
        "250-GiB reserve = PASS"
    )

    # --------------------------------------------------------
    # I. Final report
    # --------------------------------------------------------

    report = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.7",

        "part":
            "1/2",

        "status":
            "PASS_RUNTIME_BINDING_FROZEN",

        "frozen_policy": {
            "path":
                str(
                    POLICY
                ),

            "sha256":
                policy_sha_after,

            "unchanged":
                True,
        },

        "runtime_binding": {
            "path":
                str(
                    BINDING
                ),

            "sha256":
                sha256_file(
                    BINDING
                ),

            "semantic_source":
                (
                    "frozen source extraction "
                    "PASS_MARGIN_BINDING_SOURCE_EXTRACTED"
                ),

            "predictive_angular_dilation_input":
                "additional_class_margin_m",

            "symbolic_semantics":
                "m_extra_only",

            "part_a_generic_margin_reapplied":
                False,

            "total_lateral_margin_used_for_actuation":
                False,

            "total_lateral_margin_role":
                "diagnostic_reference_only",
        },

        "class_smoke":
            smoke,

        "regression": {
            "targeted_tests":
                targeted_n,

            "targeted_pass":
                True,

            "full_stage6_tests":
                full_n,

            "full_stage6_pass":
                True,
        },

        "causality": {
            "future_GT_read":
                False,

            "formal_outcomes_read":
                False,

            "development_outcomes_read":
                False,

            "policy_sweep":
                False,

            "parameter_tuning":
                False,
        },

        "immutability": {
            "class_aware_policy_unchanged":
                True,

            "blocks60_66_evidence_unchanged":
                True,

            "stage5_frozen_evidence_unchanged":
                True,
        },

        "storage": {
            "free_GiB":
                free_gib,

            "hard_reserve_GiB":
                MIN_FREE_GIB,

            "pass":
                True,
        },

        "next":
            (
                "Block6.7 Part2: frozen ADB baseline "
                "matrix + constructed-oracle/metric "
                "runtime preflight; no formal tuning"
            ),
    }

    write_json(
        REPORT,
        report,
    )

    if FAILURE_REPORT.exists():
        FAILURE_REPORT.unlink()

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.7 PART 1/2 — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "frozen class-aware policy = EXACT PASS"
    )

    print(
        "runtime margin binding     = PASS"
    )

    print(
        "predictive dilation input  = m_extra ONLY"
    )

    print(
        "Part-A generic margin      = PRESERVED ONCE"
    )

    print(
        "generic margin re-applied  = NO"
    )

    print(
        "total margin actuation     = NO"
    )

    print(
        "vehicle/ped/cyclist smoke  = PASS"
    )

    print(
        "targeted regression        =",
        f"{targeted_n} / {targeted_n} PASS",
    )

    print(
        "full Stage6 regression     =",
        f"{full_n} / {full_n} PASS",
    )

    print(
        "development outcomes       = NO"
    )

    print(
        "formal outcomes            = NO"
    )

    print(
        "policy sweep/tuning        = NO"
    )

    print(
        "upstream scientific change = NO"
    )

    print(
        "STATUS = PASS_RUNTIME_BINDING_FROZEN"
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "terminal remains open = YES"
    )


try:
    main()

except BaseException as exc:

    failure = {
        "stage":
            6,

        "block":
            "6.7",

        "part":
            "1/2",

        "status":
            "BLOCKED",

        "phase":
            "runtime_class_margin_binding",

        "exception_type":
            type(
                exc
            ).__name__,

        "reason":
            str(
                exc
            ),

        "recovery":
            (
                "Do not modify the frozen "
                "class_aware_policy.py and do not "
                "run formal evaluation. Send this "
                "BLOCKED output for targeted repair."
            ),

        "training":
            False,

        "formal_evaluation":
            False,

        "parameter_tuning":
            False,
    }

    try:
        write_json(
            FAILURE_REPORT,
            failure,
        )
    except Exception:
        pass

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.7 PART 1/2 = BLOCKED"
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

    print()
    print(
        "RECOVERY:"
    )

    print(
        "Do not modify frozen "
        "class_aware_policy.py."
    )

    print(
        "Do not run formal evaluation."
    )

    print(
        "Send this BLOCKED section only."
    )

    print(
        "training             = NO"
    )

    print(
        "formal outcomes      = NO"
    )

    print(
        "parameter tuning     = NO"
    )

    print(
        "terminal remains open= YES"
    )

# Deliberately no non-zero sys.exit().

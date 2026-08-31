from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

MODULE = (
    S6
    / "src/iscai_stage6/adb/"
      "probabilistic_full_box.py"
)

TEST = (
    S6
    / "tests/"
      "test_block64_part1_stochastic_full_box.py"
)

RUNNER = (
    S6
    / "scripts/"
      "run_block64_part1_stochastic_full_box_kernel.py"
)

CLOSURE63 = (
    S6
    / "reports/block63_final_closure.json"
)

HANDOFF63 = (
    S6
    / "artifacts/block63/"
      "block63_to_block64_handoff.json"
)

FREEZE63 = (
    S6
    / "artifacts/block63/"
      "block63_final_freeze_manifest.json"
)

POSTERIOR = (
    S6
    / "artifacts/block63/"
      "identity_safe_development_gaussian_posterior.jsonl"
)

CONTRACT = (
    S6
    / "configs/"
      "block64_part1_stochastic_full_box_kernel_contract.json"
)

REPORT = (
    S6
    / "reports/"
      "block64_part1_stochastic_full_box_kernel.json"
)


EXPECTED = {
    "closure63":
        "c451c965fa4b8a237f36358da9c1c36436a4317ee0b1501e170be2c804977d1c",

    "handoff63":
        "611f1679b83fb1273cf2d1836a03073e3c557704c9656741d46ea19f80ca18c6",

    "freeze63":
        "230b800e3889a412202306c450824152b7bff2aa6e35d12376a22d007597e8f2",

    "posterior":
        "318b22dfc3788195a1432169075d1c8654f0d183337711918db2303b5668f5ea",

    "module":
        "51daeab95713d6e7d2738ced466be4844b73dce70a09138b2b769de880462995",
}


def require(
    condition,
    message,
):
    if not bool(condition):
        raise RuntimeError(message)


def sha256_file(
    path: Path,
) -> str:
    digest = sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def write_json(
    path: Path,
    payload,
) -> str:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        +
        "\n",
        encoding="utf-8",
    )

    return sha256_file(
        path
    )


def run_tests(
    pattern,
):
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(
                S6
                / "tests"
            ),
            "-p",
            pattern,
        ],
        cwd=str(
            S6
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        process.stdout,
    )

    count = (
        int(
            match.group(1)
        )
        if match
        else None
    )

    return (
        process,
        count,
    )


try:
    print(
        "=" * 72
    )

    print(
        "BLOCK 6.4 PART1 — RESUME AFTER TEST-ONLY REPAIR"
    )

    print(
        "=" * 72
    )

    print()
    print(
        "===== D. REPAIRED DEDICATED TESTS ====="
    )

    for path, expected, label in (
        (
            CLOSURE63,
            EXPECTED["closure63"],
            "Block6.3 closure",
        ),
        (
            HANDOFF63,
            EXPECTED["handoff63"],
            "Block6.3 handoff",
        ),
        (
            FREEZE63,
            EXPECTED["freeze63"],
            "Block6.3 freeze",
        ),
        (
            POSTERIOR,
            EXPECTED["posterior"],
            "identity-safe posterior",
        ),
        (
            MODULE,
            EXPECTED["module"],
            "scientific module",
        ),
    ):
        require(
            path.is_file(),
            f"Missing {label}: {path}",
        )

        require(
            sha256_file(path)
            ==
            expected,
            (
                f"{label} SHA changed."
            ),
        )

    dedicated, dedicated_count = (
        run_tests(
            "test_block64_part1_stochastic_full_box.py"
        )
    )

    print(
        dedicated.stdout
    )

    require(
        dedicated.returncode == 0,
        (
            "Dedicated tests still fail."
        ),
    )

    require(
        dedicated_count == 14,
        (
            "Dedicated test count changed: "
            f"{dedicated_count}"
        ),
    )

    print(
        "dedicated tests = PASS | 14"
    )

    print()
    print(
        "===== E. FULL STAGE6 REGRESSION ====="
    )

    regression, regression_count = (
        run_tests(
            "test_*.py"
        )
    )

    print(
        regression.stdout
    )

    require(
        regression.returncode == 0,
        (
            "Full Stage6 regression failed."
        ),
    )

    require(
        regression_count == 112,
        (
            "Expected 112 Stage6 tests, got "
            f"{regression_count}."
        ),
    )

    print(
        "full regression = PASS | 112"
    )

    module_sha = sha256_file(
        MODULE
    )

    test_sha = sha256_file(
        TEST
    )

    runner_sha = sha256_file(
        RUNNER
    )

    print()
    print(
        "===== F. BLOCK6.4 PART1 CONTRACT ====="
    )

    contract = {
        "stage":
            6,

        "block":
            "6.4-Part1",

        "status":
            "PASS_STOCHASTIC_FULL_BOX_KERNEL_CONTRACT",

        "upstream": {
            "block63_closure_sha256":
                EXPECTED["closure63"],

            "block63_handoff_sha256":
                EXPECTED["handoff63"],

            "block63_freeze_sha256":
                EXPECTED["freeze63"],

            "identity_safe_posterior_sha256":
                EXPECTED["posterior"],
        },

        "trajectory_uncertainty": {
            "source":
                "calibrated_predictive_covariance_H0_m2",

            "mean_source":
                "mean_displacement_H0_m",

            "anchor_source":
                "latest_position_H0_m",

            "absolute_mean":
                (
                    "latest_position_H0_m + "
                    "mean_displacement_H0_m"
                ),

            "joint_sampling_semantics":
                (
                    "product_of_frozen_calibrated_"
                    "per_horizon_Gaussian_marginals"
                ),

            "cross_horizon_covariance_invented":
                False,

            "Stage5_sampler":
                "sample_product_of_horizon_gaussians",

            "sample_count":
                "RUNTIME_PARAMETER_NOT_FROZEN",

            "seed":
                "RUNTIME_PARAMETER_NOT_FROZEN",
        },

        "heading": {
            "source":
                "resolve_samplewise_headings",

            "state_provenance":
                "predicted_sample",

            "orientation_provenance":
                "predicted_tangent",

            "low_speed_behavior":
                (
                    "frozen_Stage5_previous_causal_or_"
                    "predicted_heading_carry_forward"
                ),

            "future_GT_heading":
                False,
        },

        "full_box_geometry": {
            "center":
                "sampled_absolute_H0_position",

            "length_width_height":
                "current_causal_box_held",

            "corners_per_box":
                8,

            "centroid_only":
                False,

            "projector":
                "project_box_to_headlamp",

            "communication_codebook_reused":
                False,
        },

        "test_repair": {
            "scientific_module_changed":
                False,

            "reason":
                (
                    "compare low-speed carry-forward "
                    "to stored causal Box3D yaw rather "
                    "than pre-Box3D float literal"
                ),

            "numerical_tolerance_added":
                False,
        },

        "not_yet_frozen": {
            "MC_sample_count":
                True,

            "MC_seed_set":
                True,

            "occupancy_gamma":
                True,

            "scientific_illumination_grid":
                True,

            "class_margins":
                True,

            "dimming_floors":
                True,

            "temporal_smoothing":
                True,

            "actuation_rate_limits":
                True,
        },

        "not_yet_implemented": {
            "occupancy_rasterization":
                True,

            "occupancy_thresholding":
                True,

            "multi_actor_mask_aggregation":
                True,

            "class_aware_policy":
                True,

            "temporal_smoothing":
                True,

            "actuation_rate_limits":
                True,
        },

        "scientific_execution": {
            "real_posterior_sampling":
                False,

            "formal_evaluation":
                False,

            "model_forward":
                False,

            "training":
                False,

            "recalibration":
                False,

            "parameter_tuning":
                False,
        },

        "implementation": {
            "module":
                str(MODULE),

            "module_sha256":
                module_sha,

            "test":
                str(TEST),

            "test_sha256":
                test_sha,

            "runner":
                str(RUNNER),

            "runner_sha256":
                runner_sha,
        },
    }

    contract_sha = write_json(
        CONTRACT,
        contract,
    )

    print(
        "contract =",
        CONTRACT,
    )

    print(
        "contract SHA256 =",
        contract_sha,
    )

    print()
    print(
        "===== G. BLOCK6.4 PART1 REPORT ====="
    )

    report = {
        "stage":
            6,

        "block":
            "6.4-Part1",

        "status":
            "PASS_BLOCK64_PART1_STOCHASTIC_FULL_BOX_KERNEL",

        "contract": {
            "path":
                str(CONTRACT),

            "sha256":
                contract_sha,
        },

        "implementation": {
            "module":
                str(MODULE),

            "module_sha256":
                module_sha,

            "test":
                str(TEST),

            "test_sha256":
                test_sha,

            "runner":
                str(RUNNER),

            "runner_sha256":
                runner_sha,
        },

        "tests": {
            "dedicated": {
                "status":
                    "PASS",

                "count":
                    14,
            },

            "full_stage6": {
                "status":
                    "PASS",

                "count":
                    112,
            },
        },

        "scientific_semantics": {
            "calibrated_covariance_used":
                True,

            "full_box_uncertainty_propagation":
                True,

            "corners_per_sampled_box":
                8,

            "centroid_only":
                False,

            "cross_horizon_covariance_invented":
                False,

            "samplewise_tangent_heading":
                True,

            "future_GT":
                False,

            "perfect_identity":
                False,
        },

        "scope": {
            "MC_N_frozen":
                False,

            "seed_set_frozen":
                False,

            "gamma_frozen":
                False,

            "scientific_grid_frozen":
                False,

            "real_posterior_sampled":
                False,

            "formal_evaluation":
                False,

            "training":
                False,

            "recalibration":
                False,
        },

        "next":
            (
                "BLOCK6.4_PART2_DEVELOPMENT_MC_"
                "CONVERGENCE_AND_OCCUPANCY_KERNEL"
            ),
    }

    report_sha = write_json(
        REPORT,
        report,
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "report SHA256 =",
        report_sha,
    )

    print()
    print(
        "===== H. BLOCK6.4 PART1 — FINAL ====="
    )

    print(
        "STATUS = "
        "PASS_BLOCK64_PART1_STOCHASTIC_FULL_BOX_KERNEL"
    )

    print(
        "scientific module changed      = NO"
    )

    print(
        "calibrated covariance          = USED"
    )

    print(
        "cross-horizon covariance       = NOT INVENTED"
    )

    print(
        "samplewise heading             = PREDICTED TANGENT"
    )

    print(
        "sampled geometry               = FULL BOX"
    )

    print(
        "corners per sampled box        = 8"
    )

    print(
        "centroid-only                  = NO"
    )

    print(
        "MC sample count frozen         = NO"
    )

    print(
        "seed set frozen                = NO"
    )

    print(
        "occupancy gamma frozen         = NO"
    )

    print(
        "scientific grid frozen         = NO"
    )

    print(
        "real posterior sampled         = NO"
    )

    print(
        "formal evaluation              = NO"
    )

    print(
        "training/recalibration         = NO / NO"
    )

    print(
        "dedicated tests                = PASS | 14"
    )

    print(
        "full Stage6 regression         = PASS | 112"
    )

    print(
        "module SHA256                  =",
        module_sha,
    )

    print(
        "test SHA256                    =",
        test_sha,
    )

    print(
        "runner SHA256                  =",
        runner_sha,
    )

    print(
        "contract SHA256                =",
        contract_sha,
    )

    print(
        "report SHA256                  =",
        report_sha,
    )

    print(
        "NEXT = BLOCK6.4 PART2 "
        "DEVELOPMENT MC CONVERGENCE + "
        "PROBABILISTIC OCCUPANCY KERNEL"
    )

    print(
        "terminal remains open = YES"
    )


except BaseException as exc:
    print()
    print(
        "=" * 72
    )

    print(
        "BLOCK6.4 PART1 RESUME — CONTROLLED BLOCK"
    )

    print(
        "=" * 72
    )

    print(
        "exception =",
        type(exc).__name__,
        str(exc),
    )

    traceback.print_exc(
        limit=16
    )

    print()
    print(
        "scientific module modified = NO"
    )

    print(
        "formal evaluation = NO"
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no sys.exit().

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path("/home/agni/waymo")
S4 = ROOT / "iscai_stage4"
S5 = ROOT / "iscai_stage5"
S6 = ROOT / "iscai_stage6"
S7 = ROOT / "iscai_stage7"

BLOCK72 = (
    S7
    / "configs/"
      "stage7_block72_preoutcome_experimental_protocol_freeze.json"
)

BLOCK73 = (
    S7
    / "configs/"
      "stage7_block73_exact_training_architecture_binding.json"
)

BASE_CONFIG = (
    S4
    / "configs/stage4_gaussian_gru.json"
)

RUNNER = (
    S4
    / "scripts/run_block44_gaussian_training.py"
)

DET_CKPT = (
    S4
    / "artifacts/block43/deterministic_gru.pt"
)

NORMALIZATION = (
    S4
    / "artifacts/block43/fit_normalization.json"
)

GAUSSIAN_SOURCE = (
    S4
    / "src/iscai_stage4/ml/gaussian_gru.py"
)


EXPECTED = {
    BLOCK72:
        (
            "6eb7d6f0662d35ca44a0b3975ab2fa54"
            "830bef4fc171e08370f4e0b4aaaa1c5b"
        ),

    BLOCK73:
        (
            "609071d7df6676d374b240cb1a6be0ff"
            "c98317bcf29ee9354d1e92e2bbe83b3e"
        ),

    DET_CKPT:
        (
            "5456a76b84d558e9983a59b9f1d3060b"
            "a245e0d36d60883519654f809996dbc5"
        ),

    NORMALIZATION:
        (
            "3d7fc0a66d4a4f566f6569befa9c376"
            "6ecae21830bb4e46256df2f326a82a5f6"
        ),

    GAUSSIAN_SOURCE:
        (
            "b5e206d7e62e437bd3b64a5c20c56cab"
            "6319494b119882794defeb70a9494bb8"
        ),

    RUNNER:
        (
            "6b4be296fb9d93f81d10b353a37b3427"
            "b63e437e6f123b10950db07697457218"
        ),

    S4 / "reports/stage4_final_closure.json":
        (
            "570da4feb918b1025b5e85cc919360d9"
            "22b471c468c85b3844f13fb7774e7c2f"
        ),

    S5 / "reports/stage5_final_closure.json":
        (
            "c83731948749f3ca20742aaac6b7474f"
            "53c755cbc8209dd31db969d229f60610"
        ),

    S6 / "reports/stage6_final_certificate.json":
        (
            "e3870d8c93456f0bf6470467dcbf89ee"
            "4e75e7910c6a364c487d2f30e8163cfc"
        ),

    S6 / "artifacts/stage6_to_stage7_handoff.json":
        (
            "7456e465043fa85e257fc4322159ee44c"
            "008992888aeb8d040d4cfeead9fb449"
        ),
}


TWIN_SEEDS = {
    "communication":
        1049954434,

    "adb":
        2005353450,
}


class FailClosed(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise FailClosed(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def read_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def write_json(path: Path, payload):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(
        path
    )


def under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(
            parent.resolve()
        )
        return True

    except ValueError:
        return False


def verify_frozen_anchors():
    result = {}

    for path, expected in EXPECTED.items():

        require(
            path.is_file(),
            f"Missing frozen authority: {path}",
        )

        actual = sha256_file(
            path
        )

        require(
            actual == expected,
            (
                f"Frozen authority changed: "
                f"{path}\n"
                f"expected={expected}\n"
                f"actual={actual}"
            ),
        )

        result[
            str(path)
        ] = actual

    return result


def stage7_seed_from_protocol(
    twin: str,
) -> int:

    block72 = read_json(
        BLOCK72
    )

    independent = (
        block72[
            "pdf_comparison_systems"
        ][
            "independent_models"
        ]
    )

    if twin == "communication":

        value = int(
            independent[
                "communication_predictor"
            ][
                "seed"
            ]
        )

    elif twin == "adb":

        value = int(
            independent[
                "ADB_predictor"
            ][
                "seed"
            ]
        )

    else:
        raise FailClosed(
            f"Unknown twin: {twin}"
        )

    require(
        value
        ==
        TWIN_SEEDS[twin],
        (
            "Frozen Block7.2 seed "
            "does not match bound value."
        ),
    )

    return value


def prepare_engine_config(
    twin: str,
    seed: int,
    twin_root: Path,
) -> Path:

    base = read_json(
        BASE_CONFIG
    )

    require(
        base.get("status")
        ==
        "PRETRAIN_FROZEN",
        "Stage4 Gaussian config status changed.",
    )

    require(
        base.get("purpose")
        ==
        "full_covariance_Gaussian_probabilistic_GRU",
        "Stage4 Gaussian purpose changed.",
    )

    # Stage7 independent-model protocol:
    # same frozen training envelope,
    # different preregistered seed only.
    config = copy.deepcopy(
        base
    )

    config[
        "training"
    ][
        "seed"
    ] = int(seed)

    # Fail closed: prove seed is the only scientific
    # difference from the frozen Stage4 config.
    restored = copy.deepcopy(
        config
    )

    restored[
        "training"
    ][
        "seed"
    ] = int(
        base[
            "training"
        ][
            "seed"
        ]
    )

    require(
        restored == base,
        (
            "Twin engine config differs from "
            "Stage4 Gaussian config by more "
            "than the preregistered seed."
        ),
    )

    config_path = (
        twin_root
        / "engine_config.json"
    )

    write_json(
        config_path,
        config,
    )

    return config_path


def import_frozen_runner():

    spec = (
        importlib.util
        .spec_from_file_location(
            "stage7_frozen_block44_engine",
            RUNNER,
        )
    )

    require(
        spec is not None
        and
        spec.loader is not None,
        "Could not construct frozen runner spec.",
    )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


def redirect_runner_paths(
    module,
    *,
    config_path: Path,
    twin_root: Path,
):
    """
    Redirect only Block4.4-owned mutable outputs
    into Stage7.

    Stage4 Block4.3 fit/development caches,
    deterministic checkpoint and normalization
    remain read-only upstream inputs.
    """

    artifact_root = (
        twin_root
        / "artifacts"
    )

    report_root = (
        twin_root
        / "reports"
    )

    log_root = (
        twin_root
        / "logs"
    )

    artifact_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    block44_source_artifacts = (
        S4
        / "artifacts/block44"
    ).resolve()

    stage4_reports = (
        S4
        / "reports"
    ).resolve()

    stage0_docs = (
        ROOT
        / "iscai_stage0/docs"
    ).resolve()

    redirects = {}

    for name, value in list(
        vars(module).items()
    ):

        if not isinstance(
            value,
            Path,
        ):
            continue

        old = value.resolve()

        new = None

        if old == BASE_CONFIG.resolve():

            new = config_path

        elif under(
            old,
            block44_source_artifacts,
        ):

            relative = (
                old.relative_to(
                    block44_source_artifacts
                )
            )

            new = (
                artifact_root
                / relative
            )

        elif (
            old == stage4_reports
        ):

            new = report_root

        elif (
            under(
                old,
                stage4_reports,
            )
            and
            old.name
            in (
                "block44_gaussian_gru.json",
                "block44_training_failure.json",
            )
        ):

            new = (
                report_root
                / old.name
            )

        elif (
            under(
                old,
                stage0_docs,
            )
            and
            "implementation"
            in old.name.lower()
        ):

            new = (
                log_root
                / old.name
            )

        if new is None:
            continue

        module.__dict__[
            name
        ] = new

        redirects[
            name
        ] = {
            "from":
                str(old),

            "to":
                str(
                    new.resolve()
                ),
        }

    config_redirected = any(
        item[
            "from"
        ]
        ==
        str(
            BASE_CONFIG.resolve()
        )
        for item in redirects.values()
    )

    artifact_redirected = any(
        under(
            Path(
                item[
                    "from"
                ]
            ),
            block44_source_artifacts,
        )
        for item in redirects.values()
    )

    report_redirected = any(
        (
            "/iscai_stage4/reports"
            in item[
                "from"
            ]
        )
        for item in redirects.values()
    )

    require(
        config_redirected,
        (
            "Frozen runner Gaussian config "
            "path was not redirected."
        ),
    )

    require(
        artifact_redirected,
        (
            "Frozen runner Block4.4 artifact "
            "paths were not redirected."
        ),
    )

    require(
        report_redirected,
        (
            "Frozen runner Block4.4 report "
            "path was not redirected."
        ),
    )

    return redirects


def install_project_write_guard():
    """
    Fail closed on any mutation under
    /home/agni/waymo outside iscai_stage7.

    Reads from Stage0-6 remain allowed.
    Device/cache/tmp access outside project root
    remains allowed.
    """

    root = ROOT.resolve()
    allowed = S7.resolve()

    write_bits = (
        os.O_WRONLY
        |
        os.O_RDWR
        |
        os.O_CREAT
        |
        os.O_TRUNC
        |
        os.O_APPEND
    )

    def protected_project_path(value):
        if not isinstance(
            value,
            (
                str,
                bytes,
                os.PathLike,
            ),
        ):
            return False

        try:
            path = Path(
                value
            ).resolve()

        except Exception:
            return False

        if not under(
            path,
            root,
        ):
            return False

        if under(
            path,
            allowed,
        ):
            return False

        return True

    def audit(event, args):

        if event == "open":

            if not args:
                return

            path = args[0]

            mode = (
                args[1]
                if len(args) > 1
                else None
            )

            flags = (
                args[2]
                if len(args) > 2
                else None
            )

            write_requested = False

            if isinstance(
                mode,
                str,
            ):

                write_requested = any(
                    token in mode
                    for token in (
                        "w",
                        "a",
                        "x",
                        "+",
                    )
                )

            if isinstance(
                flags,
                int,
            ):

                write_requested = (
                    write_requested
                    or
                    bool(
                        flags
                        &
                        write_bits
                    )
                )

            if (
                write_requested
                and
                protected_project_path(
                    path
                )
            ):

                raise PermissionError(
                    (
                        "STAGE7 WRITE GUARD: "
                        "upstream project mutation "
                        f"blocked: {path}"
                    )
                )

        elif event in (
            "os.remove",
            "os.unlink",
            "os.rmdir",
            "os.mkdir",
            "os.rename",
            "os.replace",
        ):

            candidates = (
                args[:2]
                if event
                in (
                    "os.rename",
                    "os.replace",
                )
                else
                args[:1]
            )

            for path in candidates:

                if protected_project_path(
                    path
                ):

                    raise PermissionError(
                        (
                            "STAGE7 WRITE GUARD: "
                            f"{event} blocked: "
                            f"{path}"
                        )
                    )

    sys.addaudithook(
        audit
    )


def locate_single(
    root: Path,
    name: str,
) -> Path:

    candidates = sorted(
        root.rglob(
            name
        )
    )

    require(
        len(candidates) == 1,
        (
            f"Expected exactly one {name} "
            f"under {root}; "
            f"found {len(candidates)}."
        ),
    )

    return candidates[0]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--twin",
        choices=(
            "communication",
            "adb",
        ),
        required=True,
    )

    args = parser.parse_args()

    twin = str(
        args.twin
    )

    print("=" * 78)
    print("STAGE 7 — BLOCK 7.4C")
    print("INDEPENDENT GAUSSIAN TRAJECTORY TWIN TRAINING")
    print("twin =", twin)
    print("=" * 78)

    before = verify_frozen_anchors()

    seed = stage7_seed_from_protocol(
        twin
    )

    twin_root = (
        S7
        / "artifacts/block74c"
        / twin
    )

    require(
        not twin_root.exists(),
        (
            "Twin output directory already exists. "
            "Refusing silent overwrite/resume:\n"
            f"{twin_root}"
        ),
    )

    twin_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    config_path = (
        prepare_engine_config(
            twin,
            seed,
            twin_root,
        )
    )

    print(
        "preregistered seed =",
        seed,
    )

    print(
        "engine config =",
        config_path,
    )

    module = import_frozen_runner()

    redirects = redirect_runner_paths(
        module,
        config_path=config_path,
        twin_root=twin_root,
    )

    print()
    print("===== ENGINE PATH REDIRECTION =====")

    for key in sorted(
        redirects
    ):

        item = redirects[
            key
        ]

        print(
            key,
            "::",
            item[
                "from"
            ],
            "->",
            item[
                "to"
            ],
        )

    # No project mutation outside Stage7
    # is possible after this point.
    install_project_write_guard()

    print()
    print("===== FROZEN BLOCK4.4 ENGINE START =====")

    result = module.main()

    if isinstance(
        result,
        int,
    ):

        require(
            result == 0,
            (
                "Frozen Block4.4 engine "
                f"returned {result}."
            ),
        )

    print()
    print("===== FROZEN BLOCK4.4 ENGINE END =====")

    checkpoint = locate_single(
        twin_root,
        "gaussian_gru.pt",
    )

    engine_report = locate_single(
        twin_root,
        "block44_gaussian_gru.json",
    )

    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    report = read_json(
        engine_report
    )

    require(
        isinstance(
            payload,
            dict,
        ),
        "Twin checkpoint is not a dictionary.",
    )

    require(
        payload.get(
            "model_type"
        )
        ==
        "GaussianTrajectoryGRU",
        "Twin checkpoint model type changed.",
    )

    require(
        payload.get(
            "calibrated"
        )
        is False,
        (
            "7.4C checkpoint must remain "
            "raw/uncalibrated."
        ),
    )

    checkpoint_config = payload.get(
        "configuration"
    )

    require(
        isinstance(
            checkpoint_config,
            dict,
        ),
        "Checkpoint configuration missing.",
    )

    require(
        int(
            checkpoint_config[
                "training"
            ][
                "seed"
            ]
        )
        ==
        seed,
        "Checkpoint seed mismatch.",
    )

    require(
        report.get(
            "status"
        )
        ==
        "PASS",
        "Frozen training engine report did not PASS.",
    )

    formal = report.get(
        "formal_validation",
        {},
    )

    require(
        formal.get(
            "used"
        )
        is False,
        (
            "Formal validation was used "
            "during twin training."
        ),
    )

    development = report.get(
        "development",
        {},
    )

    require(
        development.get(
            "NLL_improved_over_initialization"
        )
        is True,
        (
            "Twin training did not improve "
            "development NLL over initialization."
        ),
    )

    best_epoch = int(
        payload.get(
            "best_epoch",
            0,
        )
    )

    require(
        best_epoch > 0,
        (
            "Independent twin never improved "
            "beyond initialized epoch 0."
        ),
    )

    after = verify_frozen_anchors()

    require(
        before == after,
        "Frozen upstream anchors changed.",
    )

    wrapper_report = {
        "project":
            "Agni",

        "stage":
            7,

        "block":
            "7.4C",

        "status":
            "PASS_INDEPENDENT_GAUSSIAN_TWIN_TRAINED",

        "twin":
            twin,

        "role":
            (
                "independently_trained_"
                "Gaussian_trajectory_predictor"
            ),

        "preregistered_seed":
            seed,

        "training_engine": {
            "source":
                str(RUNNER),

            "source_sha256":
                sha256_file(
                    RUNNER
                ),

            "scientific_semantics":
                "exact_frozen_Block4.4_engine",
        },

        "engine_config": {
            "path":
                str(config_path),

            "sha256":
                sha256_file(
                    config_path
                ),

            "difference_from_frozen_Stage4":
                "training.seed only",
        },

        "checkpoint": {
            "path":
                str(checkpoint),

            "sha256":
                sha256_file(
                    checkpoint
                ),

            "state_dict_sha256":
                payload.get(
                    "state_dict_sha256"
                ),

            "best_epoch":
                best_epoch,

            "calibrated":
                False,
        },

        "engine_report": {
            "path":
                str(engine_report),

            "sha256":
                sha256_file(
                    engine_report
                ),

            "status":
                report.get(
                    "status"
                ),
        },

        "development": {
            "NLL_improved_over_initialization":
                development.get(
                    "NLL_improved_over_initialization"
                ),

            "initialized":
                development.get(
                    "initialized_Gaussian"
                ),

            "best_raw_uncalibrated":
                development.get(
                    "best_raw_uncalibrated"
                ),
        },

        "formal_boundary": {
            "formal_N120_opened_by_wrapper":
                False,

            "formal_GT_opened_by_wrapper":
                False,

            "formal_used_by_engine":
                False,

            "formal_model_selection":
                False,
        },

        "direct_baselines": {
            "direct_beam_training":
                "BLOCKED",

            "direct_ADB_training":
                "BLOCKED",
        },

        "upstream_anchor_sha256_before":
            before,

        "upstream_anchor_sha256_after":
            after,

        "upstream_modified":
            False,

        "post_outcome_tuning":
            False,
    }

    wrapper_path = (
        twin_root
        / "stage7_twin_training_report.json"
    )

    write_json(
        wrapper_path,
        wrapper_report,
    )

    print()
    print("=" * 78)
    print("BLOCK 7.4C TWIN STATUS = PASS")
    print("twin =", twin)
    print("seed =", seed)
    print("checkpoint =", checkpoint)
    print(
        "checkpoint SHA256 =",
        sha256_file(
            checkpoint
        ),
    )
    print(
        "state_dict SHA256 =",
        payload.get(
            "state_dict_sha256"
        ),
    )
    print(
        "best epoch =",
        best_epoch,
    )
    print("formal used = NO")
    print("Stage4/5/6 modified = NO")
    print("post-outcome tuning = NO")
    print(
        "calibration performed = NO "
        "(reserved for 7.4D)"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()

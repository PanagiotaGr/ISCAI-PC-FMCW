from pathlib import Path
import json

import numpy as np


STAGE6 = Path(
    "/home/agni/waymo/iscai_stage6/"
    "stage6_real/data/"
    "real_stage6_predictions_balanced.npz"
)

STAGE6_ORACLE = Path(
    "/home/agni/waymo/iscai_stage6/"
    "stage6_real/data/"
    "real_stage6_oracle_future_boxes.npz"
)

OUTPUT = Path(
    "stage7_real/data/"
    "stage7_common_probabilistic_input.npz"
)

REPORT = Path(
    "stage7_real/reports/"
    "stage7_common_probabilistic_input_summary.json"
)


def main():

    print("=" * 78)
    print("STAGE 7 COMMON PROBABILISTIC INPUT")
    print("=" * 78)

    d = np.load(STAGE6)
    oracle = np.load(STAGE6_ORACLE)

    required = [
        "scenario_id",
        "track_index",
        "actor_class",
        "future_mean_H_m",
        "future_std_xy_H_m",
        "future_rho_xy_H",
    ]

    missing = [
        k for k in required
        if k not in d.files
    ]

    if missing:
        raise RuntimeError(
            f"Missing required Stage-6 fields: {missing}"
        )

    scenario_id = d["scenario_id"]
    track_index = d["track_index"]
    actor_class = d["actor_class"]

    future_mean = d[
        "future_mean_H_m"
    ].astype(np.float32)

    future_std = d[
        "future_std_xy_H_m"
    ].astype(np.float32)

    future_rho = d[
        "future_rho_xy_H"
    ].astype(np.float32)

    n = len(scenario_id)

    if not (
        len(track_index)
        == len(actor_class)
        == len(future_mean)
        == len(future_std)
        == len(future_rho)
        == n
    ):
        raise RuntimeError(
            "Stage-7 common input length mismatch"
        )

    if future_mean.shape != (n, 10, 3):
        raise RuntimeError(
            f"Unexpected future_mean shape: "
            f"{future_mean.shape}"
        )

    if future_std.shape != (n, 10, 2):
        raise RuntimeError(
            f"Unexpected future_std shape: "
            f"{future_std.shape}"
        )

    if future_rho.shape != (n, 10):
        raise RuntimeError(
            f"Unexpected future_rho shape: "
            f"{future_rho.shape}"
        )

    if not np.all(np.isfinite(future_mean)):
        raise RuntimeError(
            "Non-finite future means"
        )

    if not np.all(np.isfinite(future_std)):
        raise RuntimeError(
            "Non-finite future standard deviations"
        )

    if not np.all(np.isfinite(future_rho)):
        raise RuntimeError(
            "Non-finite correlations"
        )

    if np.any(future_std <= 0):
        raise RuntimeError(
            "Non-positive Gaussian standard deviation"
        )

    if np.any(np.abs(future_rho) > 1.0):
        raise RuntimeError(
            "Invalid Gaussian correlation"
        )

    # ------------------------------------------------------
    # Align causal current geometry from the Stage-6 oracle
    # artifact by explicit scenario/track identity.
    #
    # Only CURRENT geometry is imported here. Future oracle
    # states remain evaluation-only and are not used as
    # predictor/controller input.
    # ------------------------------------------------------

    oracle_required = [
        "scenario_id",
        "track_index",
        "current_center_H_m",
    ]

    oracle_missing = [
        k
        for k in oracle_required
        if k not in oracle.files
    ]

    if oracle_missing:
        raise RuntimeError(
            "Missing oracle fields: "
            f"{oracle_missing}"
        )

    oracle_pairs = [
        (
            str(s),
            int(t),
        )
        for s, t in zip(
            oracle["scenario_id"],
            oracle["track_index"],
        )
    ]

    if len(set(oracle_pairs)) != len(oracle_pairs):
        raise RuntimeError(
            "Duplicate scenario/track pairs "
            "in Stage-6 oracle artifact"
        )

    oracle_lookup = {
        pair: i
        for i, pair in enumerate(
            oracle_pairs
        )
    }

    pairs = [
        (
            str(s),
            int(t),
        )
        for s, t in zip(
            scenario_id,
            track_index,
        )
    ]

    missing_pairs = [
        pair
        for pair in pairs
        if pair not in oracle_lookup
    ]

    if missing_pairs:
        raise RuntimeError(
            "Stage-6 prediction/oracle alignment "
            f"failed for {len(missing_pairs)} actors"
        )

    oracle_indices = np.asarray(
        [
            oracle_lookup[pair]
            for pair in pairs
        ],
        dtype=np.int64,
    )

    current_center_H_m = (
        oracle[
            "current_center_H_m"
        ][oracle_indices]
        .astype(np.float32)
    )

    if current_center_H_m.shape != (
        n,
        3,
    ):
        raise RuntimeError(
            "Unexpected current-center shape: "
            f"{current_center_H_m.shape}"
        )

    if not np.all(
        np.isfinite(
            current_center_H_m
        )
    ):
        raise RuntimeError(
            "Non-finite current actor centers"
        )

    if len(set(pairs)) != n:
        raise RuntimeError(
            "Duplicate scenario/track pairs"
        )

    unique, counts = np.unique(
        actor_class,
        return_counts=True,
    )

    class_counts = {
        str(k): int(v)
        for k, v in zip(
            unique,
            counts,
        )
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT,
        scenario_id=scenario_id,
        track_index=track_index,
        actor_class=actor_class,

        # Causal current actor geometry used by the
        # Stage-6/Stage-7 FOV cohort definition.
        current_center_H_m=
            current_center_H_m,

        # Stage-4 Gaussian prediction represented
        # in the Stage-6 headlamp coordinate frame.
        trajectory_mean_xy_H_m=
            future_mean[:, :, :2],

        trajectory_std_xy_H_m=
            future_std,

        trajectory_rho_xy_H=
            future_rho,
    )

    report = {
        "status": "PASS",
        "stage": 7,

        "source": str(STAGE6),

        "current_geometry_source":
            str(STAGE6_ORACLE),

        "current_geometry_role":
            (
                "Causal current actor center only; "
                "future oracle geometry is not used "
                "as predictor/controller input."
            ),

        "actors": int(n),

        "future_steps": 10,

        "class_counts":
            class_counts,

        "coordinate_frame":
            "Stage-6 H0 anchor headlamp frame",

        "probabilistic_representation": {
            "mean":
                "[actor, future_step, xy]",
            "std":
                "[actor, future_step, xy]",
            "rho":
                "[actor, future_step]",
        },

        "scientific_role": (
            "Common actor-level probabilistic input "
            "for Stage-7 joint communication and "
            "illumination evaluation."
        ),

        "future_used_as_predictor_input":
            False,

        "output":
            str(OUTPUT),
    }

    REPORT.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()
    print("actors =", n)
    print("class counts =", class_counts)

    print(
        "current center shape =",
        current_center_H_m.shape,
    )

    print(
        "mean shape =",
        future_mean[:, :, :2].shape,
    )

    print(
        "std shape =",
        future_std.shape,
    )

    print(
        "rho shape =",
        future_rho.shape,
    )

    print()
    print("Saved:", OUTPUT)
    print("Report:", REPORT)

    print()
    print(
        "STAGE 7 COMMON INPUT = PASS"
    )


if __name__ == "__main__":
    main()

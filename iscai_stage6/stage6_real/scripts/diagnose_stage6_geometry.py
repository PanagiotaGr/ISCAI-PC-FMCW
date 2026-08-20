from pathlib import Path
import numpy as np


PRED_PATH = Path(
    "stage6_real/data/real_stage6_predictions_balanced.npz"
)

ORACLE_PATH = Path(
    "stage6_real/data/real_stage6_oracle_future_boxes.npz"
)

FOV_DEG = 25.0
MAX_RANGE_M = 180.0


def angle_deg(xy):
    return np.degrees(
        np.arctan2(
            xy[..., 1],
            xy[..., 0],
        )
    )


def range_m(xy):
    return np.hypot(
        xy[..., 0],
        xy[..., 1],
    )


def in_fov(xy):
    a = angle_deg(xy)
    r = range_m(xy)

    return (
        (xy[..., 0] > 0.0)
        & (np.abs(a) <= FOV_DEG)
        & (r <= MAX_RANGE_M)
    )


def main():

    print("=" * 78)
    print("STAGE 6 GEOMETRY / FOV / ALIGNMENT DIAGNOSTIC")
    print("=" * 78)

    pred = np.load(PRED_PATH)
    oracle = np.load(ORACLE_PATH)

    print("\n[1] AVAILABLE KEYS")

    print("\nPrediction keys:")
    for k in pred.files:
        print(
            f"  {k:35s}",
            pred[k].shape,
            pred[k].dtype,
        )

    print("\nOracle keys:")
    for k in oracle.files:
        print(
            f"  {k:35s}",
            oracle[k].shape,
            oracle[k].dtype,
        )

    # ---------------------------------------------------------
    # Required arrays
    # ---------------------------------------------------------

    mu = pred["future_mean_H_m"]
    gt = oracle["future_center_H_m"]

    classes = pred["actor_class"]

    assert mu.shape == gt.shape
    assert mu.shape[1] == 10
    assert mu.shape[2] >= 2
    assert len(classes) == len(mu)

    print("\n[2] BASIC ALIGNMENT")

    print("actors =", len(mu))
    print("prediction shape =", mu.shape)
    print("oracle shape =", gt.shape)

    print(
        "finite predictions =",
        bool(np.isfinite(mu).all())
    )

    print(
        "finite oracle =",
        bool(np.isfinite(gt).all())
    )

    # ---------------------------------------------------------
    # Prediction error
    # ---------------------------------------------------------

    error_xy = (
        mu[..., :2]
        -
        gt[..., :2]
    )

    displacement_error = np.linalg.norm(
        error_xy,
        axis=-1,
    )

    print("\n[3] PREDICTION ↔ ORACLE POSITION ALIGNMENT")

    print(
        "mean displacement error =",
        float(displacement_error.mean()),
        "m"
    )

    print(
        "median displacement error =",
        float(np.median(displacement_error)),
        "m"
    )

    print(
        "p95 displacement error =",
        float(np.percentile(displacement_error, 95)),
        "m"
    )

    print("\nError by horizon:")

    for h in range(10):
        e = displacement_error[:, h]

        print(
            f"  h={h:02d}",
            f"mean={e.mean():.4f} m",
            f"median={np.median(e):.4f} m",
            f"p95={np.percentile(e,95):.4f} m",
        )

    # ---------------------------------------------------------
    # Angular alignment
    # ---------------------------------------------------------

    pred_angle = angle_deg(mu[..., :2])
    gt_angle = angle_deg(gt[..., :2])

    # Circular angular difference
    angle_error = (
        pred_angle
        -
        gt_angle
        +
        180.0
    ) % 360.0 - 180.0

    print("\n[4] ANGULAR ALIGNMENT")

    print(
        "mean |angular error| =",
        float(np.mean(np.abs(angle_error))),
        "deg"
    )

    print(
        "median |angular error| =",
        float(np.median(np.abs(angle_error))),
        "deg"
    )

    print(
        "p95 |angular error| =",
        float(np.percentile(np.abs(angle_error), 95)),
        "deg"
    )

    # ---------------------------------------------------------
    # FOV classification
    # ---------------------------------------------------------

    pred_fov = in_fov(mu[..., :2])
    gt_fov = in_fov(gt[..., :2])

    both = pred_fov & gt_fov
    pred_only = pred_fov & (~gt_fov)
    gt_only = (~pred_fov) & gt_fov
    neither = (~pred_fov) & (~gt_fov)

    nstates = pred_fov.size

    print("\n[5] FOV AGREEMENT")

    print(
        "total states =",
        nstates
    )

    print(
        "prediction in FOV =",
        int(pred_fov.sum()),
        f"({100*pred_fov.mean():.3f}%)"
    )

    print(
        "oracle in FOV =",
        int(gt_fov.sum()),
        f"({100*gt_fov.mean():.3f}%)"
    )

    print(
        "both in FOV =",
        int(both.sum()),
        f"({100*both.mean():.3f}%)"
    )

    print(
        "prediction only =",
        int(pred_only.sum()),
        f"({100*pred_only.mean():.3f}%)"
    )

    print(
        "oracle only =",
        int(gt_only.sum()),
        f"({100*gt_only.mean():.3f}%)"
    )

    print(
        "neither =",
        int(neither.sum()),
        f"({100*neither.mean():.3f}%)"
    )

    fov_agreement = np.mean(
        pred_fov == gt_fov
    )

    print(
        "FOV classification agreement =",
        f"{100*fov_agreement:.4f}%"
    )

    # ---------------------------------------------------------
    # Current position audit
    # ---------------------------------------------------------

    print("\n[6] CURRENT CAUSAL POSITION")

    if "current_center_H_m" not in oracle.files:

        print(
            "WARNING: current_center_H_m "
            "not present in oracle artifact"
        )

        current_fov = None

    else:

        current = oracle[
            "current_center_H_m"
        ]

        ca = angle_deg(
            current[..., :2]
        )

        cr = range_m(
            current[..., :2]
        )

        current_fov = in_fov(
            current[..., :2]
        )

        print(
            "current position shape =",
            current.shape
        )

        print(
            "current x min/max =",
            float(current[:,0].min()),
            float(current[:,0].max())
        )

        print(
            "current y min/max =",
            float(current[:,1].min()),
            float(current[:,1].max())
        )

        print(
            "current angle min/max =",
            float(ca.min()),
            float(ca.max())
        )

        print(
            "current range min/max =",
            float(cr.min()),
            float(cr.max())
        )

        print(
            "current in-FOV actors =",
            int(current_fov.sum()),
            "/",
            len(current_fov),
        )

    # ---------------------------------------------------------
    # Class analysis
    # ---------------------------------------------------------

    print("\n[7] CLASS ANALYSIS")

    for cls in np.unique(classes):

        m = classes == cls

        print("\n", cls)

        print(
            " actors =",
            int(m.sum())
        )

        print(
            " pred future states in FOV =",
            f"{100*pred_fov[m].mean():.3f}%"
        )

        print(
            " oracle future states in FOV =",
            f"{100*gt_fov[m].mean():.3f}%"
        )

        print(
            " mean position error =",
            float(
                displacement_error[m].mean()
            ),
            "m"
        )

        print(
            " mean |angular error| =",
            float(
                np.abs(
                    angle_error[m]
                ).mean()
            ),
            "deg"
        )

        if current_fov is not None:

            print(
                " current in-FOV actors =",
                int(
                    np.sum(
                        current_fov[m]
                    )
                ),
                "/",
                int(m.sum())
            )

    # ---------------------------------------------------------
    # Current-FOV → future behaviour
    # ---------------------------------------------------------

    if current_fov is not None:

        print(
            "\n[8] CURRENT-FOV ACTORS THROUGH FUTURE HORIZONS"
        )

        for h in range(10):

            m = current_fov

            print(
                f"h={h:02d}",
                f"actors={m.sum()}",
                f"predFOV={100*pred_fov[m,h].mean():.2f}%",
                f"oracleFOV={100*gt_fov[m,h].mean():.2f}%",
                f"meanErr={displacement_error[m,h].mean():.3f}m",
                f"angErr={np.abs(angle_error[m,h]).mean():.3f}deg",
            )

    # ---------------------------------------------------------
    # Predicted future entry
    # ---------------------------------------------------------

    if current_fov is not None:

        predicted_entry = (
            (~current_fov)
            &
            np.any(
                pred_fov,
                axis=1,
            )
        )

        actual_entry = (
            (~current_fov)
            &
            np.any(
                gt_fov,
                axis=1,
            )
        )

        correct_entry = (
            predicted_entry
            &
            actual_entry
        )

        false_entry = (
            predicted_entry
            &
            (~actual_entry)
        )

        missed_entry = (
            (~predicted_entry)
            &
            actual_entry
        )

        print("\n[9] FUTURE-ENTRY DIAGNOSTIC")

        print(
            "predicted entries =",
            int(predicted_entry.sum())
        )

        print(
            "actual oracle entries =",
            int(actual_entry.sum())
        )

        print(
            "correct predicted entries =",
            int(correct_entry.sum())
        )

        print(
            "false predicted entries =",
            int(false_entry.sum())
        )

        print(
            "missed oracle entries =",
            int(missed_entry.sum())
        )

    # ---------------------------------------------------------
    # Representative states
    # ---------------------------------------------------------

    print("\n[10] REPRESENTATIVE STATES")

    categories = {
        "both_in_fov": both,
        "prediction_only": pred_only,
        "oracle_only": gt_only,
        "neither": neither,
    }

    for name, mask in categories.items():

        idx = np.argwhere(mask)

        print()
        print(name)

        if len(idx) == 0:
            print("  NONE")
            continue

        actor, horizon = idx[0]

        print(
            " actor =",
            int(actor)
        )

        print(
            " horizon =",
            int(horizon)
        )

        print(
            " class =",
            classes[actor]
        )

        print(
            " pred xy =",
            mu[
                actor,
                horizon,
                :2
            ].tolist()
        )

        print(
            " oracle xy =",
            gt[
                actor,
                horizon,
                :2
            ].tolist()
        )

        print(
            " pred angle =",
            float(
                pred_angle[
                    actor,
                    horizon
                ]
            )
        )

        print(
            " oracle angle =",
            float(
                gt_angle[
                    actor,
                    horizon
                ]
            )
        )

        print(
            " position error =",
            float(
                displacement_error[
                    actor,
                    horizon
                ]
            )
        )

    # ---------------------------------------------------------
    # Diagnostic interpretation
    # ---------------------------------------------------------

    print()
    print("=" * 78)
    print("AUTOMATIC DIAGNOSTIC")
    print("=" * 78)

    problems = []

    if not np.isfinite(mu).all():
        problems.append(
            "Non-finite prediction coordinates."
        )

    if not np.isfinite(gt).all():
        problems.append(
            "Non-finite oracle coordinates."
        )

    if fov_agreement < 0.90:
        problems.append(
            "Low prediction/oracle FOV agreement."
        )

    median_angle_error = float(
        np.median(
            np.abs(
                angle_error
            )
        )
    )

    if median_angle_error > 5.0:
        problems.append(
            "Large median angular mismatch; "
            "possible frame/alignment problem."
        )

    if current_fov is not None:

        frac = float(
            current_fov.mean()
        )

        if frac < 0.5:
            print()
            print(
                "NOTE: Most balanced actors are outside "
                "the current forward ADB FOV."
            )

            print(
                "This is a DATASET/EVALUATION-POPULATION "
                "property, not by itself a frame bug."
            )

    if problems:

        print()
        print("Potential technical problems:")

        for p in problems:
            print(" -", p)

        print()
        print(
            "STATUS = INVESTIGATE"
        )

    else:

        print()
        print(
            "No strong evidence of coordinate-frame "
            "or prediction/oracle alignment failure."
        )

        print(
            "STATUS = GEOMETRY_ALIGNMENT_PASS"
        )

    print("=" * 78)


if __name__ == "__main__":
    main()

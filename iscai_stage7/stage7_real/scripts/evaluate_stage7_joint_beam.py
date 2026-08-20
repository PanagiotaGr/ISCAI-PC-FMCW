from pathlib import Path
import json
import numpy as np


INPUT = Path(
    "stage7_real/data/"
    "stage7_common_probabilistic_input.npz"
)

COHORT = Path(
    "stage7_real/data/"
    "stage7_joint_cohort.npz"
)

OUTPUT = Path(
    "stage7_real/reports/"
    "stage7_joint_beam_evaluation.json"
)

MC_SAMPLES = 500

CODEBOOKS = [
    16,
    32,
    64,
]

TARGETS = [
    0.90,
    0.95,
    0.975,
    0.99,
]

FIXED_K = [
    1,
    3,
    5,
]

SEED = 42


def sample_gaussian(
    mu,
    std,
    rho,
    n,
    rng,
):
    mux = float(mu[0])
    muy = float(mu[1])

    sx = float(std[0])
    sy = float(std[1])

    r = float(
        np.clip(
            rho,
            -0.999,
            0.999,
        )
    )

    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)

    xs = (
        mux
        +
        sx * z1
    )

    ys = (
        muy
        +
        sy
        *
        (
            r * z1
            +
            np.sqrt(
                1.0-r*r
            )
            *
            z2
        )
    )

    return xs, ys


def beam_probabilities(
    angles,
    num_beams,
):
    normalized = (
        angles + np.pi
    ) / (
        2.0 * np.pi
    )

    indices = np.floor(
        normalized
        *
        num_beams
    ).astype(
        np.int64
    )

    indices = np.clip(
        indices,
        0,
        num_beams - 1,
    )

    counts = np.bincount(
        indices,
        minlength=num_beams,
    ).astype(
        np.float64
    )

    total = counts.sum()

    if total > 0:
        counts /= total

    return counts


def adaptive_topk(
    probabilities,
    target_mass,
):
    ranked = np.argsort(
        probabilities
    )[::-1]

    selected = []
    accumulated = 0.0

    for beam in ranked:
        selected.append(
            int(beam)
        )

        accumulated += float(
            probabilities[beam]
        )

        if accumulated >= target_mass:
            break

    return (
        selected,
        accumulated,
    )


def fixed_topk(
    probabilities,
    k,
):
    ranked = np.argsort(
        probabilities
    )[::-1]

    selected = ranked[:k]

    mass = float(
        probabilities[
            selected
        ].sum()
    )

    return (
        [
            int(x)
            for x in selected
        ],
        mass,
    )


def summary(x):
    a = np.asarray(
        x,
        dtype=np.float64,
    )

    return {
        "count":
            int(len(a)),

        "mean":
            float(a.mean()),

        "p50":
            float(
                np.percentile(
                    a,
                    50,
                )
            ),

        "p90":
            float(
                np.percentile(
                    a,
                    90,
                )
            ),

        "p95":
            float(
                np.percentile(
                    a,
                    95,
                )
            ),

        "p99":
            float(
                np.percentile(
                    a,
                    99,
                )
            ),
    }


def main():

    print("=" * 82)
    print("STAGE 7 JOINT-COHORT BEAM EVALUATION")
    print("=" * 82)

    d = np.load(INPUT)
    cohort = np.load(COHORT)

    joint_relevant = cohort[
        "joint_relevant"
    ].astype(bool)

    if len(joint_relevant) != len(
        d["actor_class"]
    ):
        raise RuntimeError(
            "Joint-cohort length mismatch"
        )

    selected_indices = np.flatnonzero(
        joint_relevant
    )

    classes = d[
        "actor_class"
    ][selected_indices]

    mu = d[
        "trajectory_mean_xy_H_m"
    ][selected_indices]

    std = d[
        "trajectory_std_xy_H_m"
    ][selected_indices]

    rho = d[
        "trajectory_rho_xy_H"
    ][selected_indices]

    rng = np.random.default_rng(
        SEED
    )

    adaptive = {}
    fixed = {}

    for nb in CODEBOOKS:

        for target in TARGETS:
            adaptive[
                (nb, target)
            ] = {
                "k": [],
                "mass": [],
            }

        for k in FIXED_K:
            fixed[
                (nb, k)
            ] = {
                "mass": [],
            }

    by_class = {
        cls: {
            "adaptive": {},
            "fixed": {},
        }
        for cls in np.unique(classes)
    }

    for cls in by_class:
        for nb in CODEBOOKS:
            for target in TARGETS:
                by_class[
                    cls
                ][
                    "adaptive"
                ][
                    (nb, target)
                ] = {
                    "k": [],
                    "mass": [],
                }

            for k in FIXED_K:
                by_class[
                    cls
                ][
                    "fixed"
                ][
                    (nb, k)
                ] = {
                    "mass": [],
                }

    n, T, _ = mu.shape

    for actor in range(n):

        cls = str(
            classes[actor]
        )

        for t in range(T):

            xs, ys = (
                sample_gaussian(
                    mu[
                        actor,
                        t,
                    ],
                    std[
                        actor,
                        t,
                    ],
                    rho[
                        actor,
                        t,
                    ],
                    MC_SAMPLES,
                    rng,
                )
            )

            angles = np.arctan2(
                ys,
                xs,
            )

            for nb in CODEBOOKS:

                probabilities = (
                    beam_probabilities(
                        angles,
                        nb,
                    )
                )

                for target in TARGETS:

                    selected, mass = (
                        adaptive_topk(
                            probabilities,
                            target,
                        )
                    )

                    adaptive[
                        (nb, target)
                    ]["k"].append(
                        len(selected)
                    )

                    adaptive[
                        (nb, target)
                    ]["mass"].append(
                        mass
                    )

                    by_class[
                        cls
                    ][
                        "adaptive"
                    ][
                        (nb, target)
                    ]["k"].append(
                        len(selected)
                    )

                    by_class[
                        cls
                    ][
                        "adaptive"
                    ][
                        (nb, target)
                    ]["mass"].append(
                        mass
                    )

                for k in FIXED_K:

                    _, mass = (
                        fixed_topk(
                            probabilities,
                            k,
                        )
                    )

                    fixed[
                        (nb, k)
                    ]["mass"].append(
                        mass
                    )

                    by_class[
                        cls
                    ][
                        "fixed"
                    ][
                        (nb, k)
                    ]["mass"].append(
                        mass
                    )

        if (
            (actor + 1) % 50 == 0
            or actor + 1 == n
        ):
            print(
                f"actors={actor+1}/{n}",
                flush=True,
            )

    adaptive_rows = []

    for nb in CODEBOOKS:
        for target in TARGETS:

            x = adaptive[
                (nb, target)
            ]

            adaptive_rows.append(
                {
                    "num_beams":
                        nb,

                    "beam_width_deg":
                        360.0 / nb,

                    "target_mass":
                        target,

                    "selected_k":
                        summary(
                            x["k"]
                        ),

                    "covered_mass":
                        summary(
                            x["mass"]
                        ),

                    "mean_overhead_fraction":
                        float(
                            np.mean(
                                x["k"]
                            )
                            /
                            nb
                        ),
                }
            )

    fixed_rows = []

    for nb in CODEBOOKS:
        for k in FIXED_K:

            x = fixed[
                (nb, k)
            ]

            fixed_rows.append(
                {
                    "num_beams":
                        nb,

                    "beam_width_deg":
                        360.0 / nb,

                    "k":
                        k,

                    "covered_mass":
                        summary(
                            x["mass"]
                        ),

                    "overhead_fraction":
                        float(
                            k / nb
                        ),
                }
            )

    class_results = {}

    for cls in sorted(
        by_class
    ):

        adaptive_rows_cls = []
        fixed_rows_cls = []

        for nb in CODEBOOKS:

            for target in TARGETS:

                x = (
                    by_class[
                        cls
                    ][
                        "adaptive"
                    ][
                        (nb, target)
                    ]
                )

                adaptive_rows_cls.append(
                    {
                        "num_beams":
                            nb,

                        "target_mass":
                            target,

                        "selected_k":
                            summary(
                                x["k"]
                            ),

                        "covered_mass":
                            summary(
                                x["mass"]
                            ),

                        "mean_overhead_fraction":
                            float(
                                np.mean(
                                    x["k"]
                                )
                                /
                                nb
                            ),
                    }
                )

            for k in FIXED_K:

                x = (
                    by_class[
                        cls
                    ][
                        "fixed"
                    ][
                        (nb, k)
                    ]
                )

                fixed_rows_cls.append(
                    {
                        "num_beams":
                            nb,

                        "k":
                            k,

                        "covered_mass":
                            summary(
                                x["mass"]
                            ),

                        "overhead_fraction":
                            float(
                                k / nb
                            ),
                    }
                )

        class_results[
            cls
        ] = {
            "adaptive_topk":
                adaptive_rows_cls,

            "fixed_k":
                fixed_rows_cls,
        }

    report = {
        "status":
            "PASS",

        "stage":
            7,

        "evaluation":
            "joint_relevant_cohort_beam_management",

        "source":
            str(INPUT),

        "cohort_source":
            str(COHORT),

        "actors":
            int(n),

        "future_steps":
            int(T),

        "evaluated_future_points":
            int(n * T),

        "mc_samples_per_future_point":
            MC_SAMPLES,

        "seed":
            SEED,

        "codebooks":
            CODEBOOKS,

        "targets":
            TARGETS,

        "fixed_k":
            FIXED_K,

        "future_used_as_input":
            False,

        "shared_posterior":
            True,

        "adaptive_topk":
            adaptive_rows,

        "fixed_k_baselines":
            fixed_rows,

        "results_by_class":
            class_results,

        "scientific_scope": (
            "Stage-5 Gaussian beam-probability and Top-K "
            "logic evaluated only on the exact Stage-6 "
            "joint-relevant actor cohort."
        ),
    }

    OUTPUT.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()
    print(
        "future points =",
        n * T,
    )

    print(
        "MC samples / point =",
        MC_SAMPLES,
    )

    print()
    print("Saved:", OUTPUT)

    print()
    print(
        "STAGE 7 JOINT BEAM EVALUATION = PASS"
    )


if __name__ == "__main__":
    main()

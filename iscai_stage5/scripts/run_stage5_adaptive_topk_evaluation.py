from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


SOURCE = Path(
    "artifacts/stage5_class_aware_trajectories.json"
)

REPORT = Path(
    "reports/stage5_adaptive_topk_evaluation.json"
)


MODELS = (
    "CV",
    "CA",
    "CTRV",
    "KALMAN",
)


CODEBOOK_SIZES = (
    16,
    32,
    64,
)


COVERAGE_TARGETS = (
    0.90,
    0.95,
    0.975,
    0.99,
)


FIXED_K_BASELINES = (
    1,
    3,
    5,
)


def beam_index(
    angle: float,
    number_of_beams: int,
) -> int:

    normalized = (
        angle + math.pi
    ) / (
        2.0 * math.pi
    )

    index = int(
        normalized
        *
        number_of_beams
    )

    return max(
        0,
        min(
            number_of_beams - 1,
            index,
        ),
    )


def build_probabilities(
    angles,
    number_of_beams,
):

    scores = [
        0.0
        for _ in range(
            number_of_beams
        )
    ]

    for angle in angles:

        index = beam_index(
            angle,
            number_of_beams,
        )

        scores[index] += 1.0


    total = sum(scores)

    if total <= 0.0:
        return None


    probabilities = [
        score / total
        for score in scores
    ]


    return probabilities


def adaptive_topk(
    probabilities,
    requested_mass,
):

    ranked = sorted(
        range(
            len(probabilities)
        ),
        key=lambda i:
            probabilities[i],
        reverse=True,
    )

    selected = []

    accumulated = 0.0


    for beam in ranked:

        selected.append(
            beam
        )

        accumulated += (
            probabilities[beam]
        )

        if (
            accumulated
            >=
            requested_mass
        ):
            break


    return (
        selected,
        accumulated,
    )


def fixed_topk(
    probabilities,
    k,
):

    ranked = sorted(
        range(
            len(probabilities)
        ),
        key=lambda i:
            probabilities[i],
        reverse=True,
    )

    selected = ranked[:k]

    covered_mass = sum(
        probabilities[i]
        for i in selected
    )

    return (
        selected,
        covered_mass,
    )


def mean(values):

    if not values:
        return 0.0

    return (
        sum(values)
        /
        len(values)
    )


def main():

    if not SOURCE.exists():

        raise RuntimeError(
            f"Missing source: {SOURCE}"
        )


    data = json.loads(
        SOURCE.read_text()
    )


    groups = defaultdict(dict)


    for item in data:

        key = (
            item["scenario_id"],
            int(
                item["track_index"]
            ),
            item["actor_type"],
        )

        model = item.get(
            "model_name",
            "UNKNOWN",
        )

        if model in MODELS:

            groups[key][model] = item


    results = {}


    valid_actors = 0


    for codebook_size in CODEBOOK_SIZES:

        codebook_result = {

            "adaptive": {},

            "fixed": {},
        }


        adaptive_stats = {

            target: {
                "K": [],
                "mass": [],
                "overhead": [],
                "success": 0,
            }

            for target
            in COVERAGE_TARGETS
        }


        fixed_stats = {

            k: {
                "mass": [],
                "overhead": [],
            }

            for k
            in FIXED_K_BASELINES
        }


        actors_this_codebook = 0


        for (
            key,
            model_records,
        ) in groups.items():


            if not all(
                model in model_records
                for model in MODELS
            ):

                continue


            angles = []


            for model in MODELS:

                trajectory = (
                    model_records[
                        model
                    ].get(
                        "trajectory",
                        [],
                    )
                )


                if len(trajectory) < 2:
                    break


                point = trajectory[1]

                x = float(
                    point[0]
                )

                y = float(
                    point[1]
                )


                angles.append(
                    math.atan2(
                        y,
                        x,
                    )
                )


            if (
                len(angles)
                !=
                len(MODELS)
            ):

                continue


            probabilities = (
                build_probabilities(
                    angles,
                    codebook_size,
                )
            )


            if probabilities is None:
                continue


            if not math.isclose(
                sum(probabilities),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):

                raise RuntimeError(
                    "Invalid probability distribution."
                )


            actors_this_codebook += 1


            for target in COVERAGE_TARGETS:

                selected, mass = (
                    adaptive_topk(
                        probabilities,
                        target,
                    )
                )


                k = len(
                    selected
                )


                overhead = (
                    k
                    /
                    codebook_size
                )


                adaptive_stats[
                    target
                ]["K"].append(
                    k
                )

                adaptive_stats[
                    target
                ]["mass"].append(
                    mass
                )

                adaptive_stats[
                    target
                ]["overhead"].append(
                    overhead
                )


                if mass >= target:

                    adaptive_stats[
                        target
                    ]["success"] += 1


            for k in FIXED_K_BASELINES:

                selected, mass = (
                    fixed_topk(
                        probabilities,
                        k,
                    )
                )


                fixed_stats[
                    k
                ]["mass"].append(
                    mass
                )

                fixed_stats[
                    k
                ]["overhead"].append(
                    len(selected)
                    /
                    codebook_size
                )


        valid_actors = max(
            valid_actors,
            actors_this_codebook,
        )


        for target in COVERAGE_TARGETS:

            stats = (
                adaptive_stats[
                    target
                ]
            )


            count = len(
                stats["K"]
            )


            codebook_result[
                "adaptive"
            ][
                str(target)
            ] = {

                "requested_mass":
                    target,

                "actors":
                    count,

                "mean_K":
                    mean(
                        stats["K"]
                    ),

                "min_K":
                    min(
                        stats["K"]
                    )
                    if stats["K"]
                    else 0,

                "max_K":
                    max(
                        stats["K"]
                    )
                    if stats["K"]
                    else 0,

                "mean_selected_mass":
                    mean(
                        stats["mass"]
                    ),

                "empirical_target_success":
                    (
                        stats["success"]
                        /
                        count
                    )
                    if count
                    else 0.0,

                "mean_overhead_fraction":
                    mean(
                        stats[
                            "overhead"
                        ]
                    ),

                "overhead_reduction_vs_exhaustive":
                    1.0
                    -
                    mean(
                        stats[
                            "overhead"
                        ]
                    ),
            }


        for k in FIXED_K_BASELINES:

            stats = fixed_stats[k]


            codebook_result[
                "fixed"
            ][
                str(k)
            ] = {

                "K":
                    k,

                "actors":
                    len(
                        stats["mass"]
                    ),

                "mean_selected_mass":
                    mean(
                        stats["mass"]
                    ),

                "mean_overhead_fraction":
                    mean(
                        stats[
                            "overhead"
                        ]
                    ),

                "overhead_reduction_vs_exhaustive":
                    1.0
                    -
                    mean(
                        stats[
                            "overhead"
                        ]
                    ),
            }


        results[
            str(codebook_size)
        ] = codebook_result


    payload = {

        "source":
            str(SOURCE),

        "models":
            list(MODELS),

        "actors":
            valid_actors,

        "codebook_sizes":
            list(
                CODEBOOK_SIZES
            ),

        "coverage_targets":
            list(
                COVERAGE_TARGETS
            ),

        "fixed_k_baselines":
            list(
                FIXED_K_BASELINES
            ),

        "future_used":
            False,

        "results":
            results,
    }


    sha = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        ).encode()
    ).hexdigest()


    status = "PASS"


    for codebook in results.values():

        for item in (
            codebook[
                "adaptive"
            ].values()
        ):

            if (
                item[
                    "actors"
                ]
                <= 0
            ):

                status = "FAIL"


            if (
                item[
                    "empirical_target_success"
                ]
                < 1.0
            ):

                status = "FAIL"


            if not (
                0.0
                <
                item[
                    "mean_overhead_fraction"
                ]
                <=
                1.0
            ):

                status = "FAIL"


    report = {

        **payload,

        "sha256":
            sha,

        "status":
            status,
    }


    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    REPORT.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
    )


    print(
        "===== Stage5 Adaptive Top-K Evaluation ====="
    )

    print(
        "actors =",
        valid_actors,
    )


    for size in CODEBOOK_SIZES:

        print(
            f"\nCODEBOOK = {size}"
        )


        adaptive = (
            results[
                str(size)
            ]["adaptive"]
        )


        for target in COVERAGE_TARGETS:

            item = adaptive[
                str(target)
            ]

            print(
                f"target={target:.3f} "
                f"mean_K={item['mean_K']:.3f} "
                f"mass={item['mean_selected_mass']:.6f} "
                f"overhead={item['mean_overhead_fraction']:.6f} "
                f"reduction={item['overhead_reduction_vs_exhaustive']:.6f}"
            )


    print(
        "\nfuture_used = NO"
    )

    print(
        "SHA256 =",
        sha,
    )

    print(
        "STATUS =",
        status,
    )

    print(
        "report =",
        REPORT,
    )


    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

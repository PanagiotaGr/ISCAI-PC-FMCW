from pathlib import Path

src = Path("scripts/run_stage4_reduced_cuda.py")
dst = Path("scripts/run_stage4_parallel_cuda.py")

text = src.read_text()

# --------------------------------------------------
# 1. argparse
# --------------------------------------------------

if "import argparse" not in text:
    text = text.replace(
        "import json\n",
        "import argparse\nimport json\n",
        1,
    )

# --------------------------------------------------
# 2. Fewer DataLoader workers per training process
# --------------------------------------------------

text = text.replace(
    "NUM_WORKERS = 8",
    "NUM_WORKERS = 4",
)

# --------------------------------------------------
# 3. Add argument parser before main()
# --------------------------------------------------

marker = """# ============================================================
# MAIN
# ============================================================

def main():
"""

replacement = """# ============================================================
# PARALLEL WORKER ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--worker",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--workers",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError(
            "--workers must be >= 1"
        )

    if not (
        0 <= args.worker < args.workers
    ):
        raise ValueError(
            "--worker must satisfy "
            "0 <= worker < workers"
        )

    return args


# ============================================================
# MAIN
# ============================================================

def main():
"""

if marker not in text:
    raise RuntimeError(
        "Could not find MAIN marker"
    )

text = text.replace(
    marker,
    replacement,
    1,
)

# --------------------------------------------------
# 4. Split configurations between workers
# --------------------------------------------------

old = """def main():

    configs = build_configs()
"""

new = """def main():

    args = parse_args()

    all_configs = build_configs()

    configs = [
        cfg
        for index, cfg in enumerate(all_configs)
        if index % args.workers == args.worker
    ]

    worker_results_file = (
        REPORT
        /
        f"worker_{args.worker}_results.json"
    )

    print(
        "\\n========================================"
    )

    print(
        "PARALLEL STAGE 4 WORKER"
    )

    print(
        "WORKER:",
        args.worker,
        "/",
        args.workers,
    )

    print(
        "GLOBAL CONFIGS:",
        len(all_configs),
    )

    print(
        "WORKER CONFIGS:",
        len(configs),
    )

    print(
        "WORKER RESULTS:",
        worker_results_file,
    )

    print(
        "========================================\\n"
    )
"""

if old not in text:
    raise RuntimeError(
        "Could not find main config block"
    )

text = text.replace(
    old,
    new,
    1,
)

# --------------------------------------------------
# 5. Load worker-specific previous results
# --------------------------------------------------

old = """    all_results = (
        load_existing_results()
    )


    completed_names = {
"""

new = """    all_results = (
        load_existing_results()
    )

    worker_results = []

    if worker_results_file.exists():

        with open(
            worker_results_file,
            "r",
        ) as f:

            worker_results = json.load(f)

        if not isinstance(
            worker_results,
            list,
        ):

            raise RuntimeError(
                "Worker results file is not a list."
            )

        canonical_names = {
            r["name"]
            for r in all_results
            if (
                isinstance(r, dict)
                and
                "name" in r
            )
        }

        for result in worker_results:

            if (
                isinstance(result, dict)
                and
                "name" in result
                and
                result["name"] not in canonical_names
            ):

                all_results.append(
                    result
                )

                canonical_names.add(
                    result["name"]
                )


    completed_names = {
"""

if old not in text:
    raise RuntimeError(
        "Could not find result loading block"
    )

text = text.replace(
    old,
    new,
    1,
)

# --------------------------------------------------
# 6. Worker-specific save after each experiment
# --------------------------------------------------

old = """            all_results.append(
                result
            )


            completed_names.add(
                name
            )


            # ------------------------------------------------
            # Save immediately.
            #
            # Therefore if the process is interrupted later,
            # every fully completed experiment is retained.
            # ------------------------------------------------

            atomic_save_json(
                all_results,
                RESULTS_FILE,
            )


            best_models = (
                calculate_best_models(
                    all_results
                )
            )


            atomic_save_json(
                best_models,
                BEST_FILE,
            )
"""

new = """            all_results.append(
                result
            )

            worker_results.append(
                result
            )

            completed_names.add(
                name
            )


            # ------------------------------------------------
            # IMPORTANT:
            # Parallel workers NEVER write to the shared
            # all_results.json or best_models.json.
            #
            # Each worker writes only to its own file.
            # ------------------------------------------------

            atomic_save_json(
                worker_results,
                worker_results_file,
            )
"""

if old not in text:
    raise RuntimeError(
        "Could not find experiment save block"
    )

text = text.replace(
    old,
    new,
    1,
)

# --------------------------------------------------
# 7. Remove shared final save
# --------------------------------------------------

old = """    atomic_save_json(
        all_results,
        RESULTS_FILE,
    )


    atomic_save_json(
        best_models,
        BEST_FILE,
    )
"""

new = """    atomic_save_json(
        worker_results,
        worker_results_file,
    )
"""

if old not in text:
    raise RuntimeError(
        "Could not find final shared save block"
    )

text = text.replace(
    old,
    new,
    1,
)

dst.write_text(text)

print("CREATED:", dst)

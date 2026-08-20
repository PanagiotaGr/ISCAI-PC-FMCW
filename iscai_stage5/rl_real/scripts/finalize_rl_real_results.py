from pathlib import Path
import json
from datetime import datetime, timezone


TRAINING_REPORT = Path(
    "rl_real/reports/"
    "dqn_horizon_training_report.json"
)

COMPARISON_REPORT = Path(
    "rl_real/reports/"
    "dqn_vs_baselines.json"
)

STATE_SUMMARY = Path(
    "rl_real/reports/"
    "real_rl_states_summary.json"
)

OUTPUT = Path(
    "rl_real/reports/"
    "rl_final_real_results.json"
)


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r") as f:
        return json.load(f)


def main():

    training = load_json(
        TRAINING_REPORT
    )

    comparison = load_json(
        COMPARISON_REPORT
    )

    states = load_json(
        STATE_SUMMARY
    )

    results = comparison[
        "results"
    ]

    dqn = results[
        "DQN"
    ]

    adaptive = results[
        "AdaptiveTopK95"
    ]

    fixed1 = results[
        "FixedK1"
    ]

    fixed3 = results[
        "FixedK3"
    ]

    fixed5 = results[
        "FixedK5"
    ]

    # --------------------------------------------------------
    # Determine best controller under the exact reward used
    # in the common held-out evaluation.
    # --------------------------------------------------------

    ranked = sorted(
        results.items(),
        key=lambda item:
            item[1]["mean_reward"],
        reverse=True,
    )

    best_controller = ranked[0][0]

    adaptive_beats_dqn = (
        adaptive["mean_reward"]
        >
        dqn["mean_reward"]
    )

    # --------------------------------------------------------
    # Final package
    # --------------------------------------------------------

    report = {
        "stage":
            "stage5_rl_extension",

        "status":
            "COMPLETE",

        "scope":
            (
                "Exploratory horizon-sequential RL extension "
                "using real WOMD-derived Gaussian beam posteriors."
            ),

        "scientific_scope_note":
            (
                "The 10 sequential states correspond to forecast "
                "horizons from one causal Gaussian prediction. "
                "They are not 10 newly observed sensor frames; "
                "therefore this experiment is not claimed as "
                "multi-frame closed-loop RL."
            ),

        "future_used_as_input":
            False,

        "rl_dataset": {
            "source":
                str(STATE_SUMMARY),

            "data_source":
                states.get(
                    "data_source"
                ),

            "processed_trajectories":
                states.get(
                    "processed_trajectories"
                ),

            "states":
                states.get(
                    "states"
                ),

            "num_beams":
                states.get(
                    "num_beams"
                ),

            "monte_carlo_samples_per_state":
                states.get(
                    "monte_carlo_samples_per_state"
                ),

            "posterior_shape":
                states.get(
                    "posterior_shape"
                ),
        },

        "dqn_training": {
            "source":
                str(TRAINING_REPORT),

            "algorithm":
                training.get(
                    "algorithm"
                ),

            "state":
                training.get(
                    "state"
                ),

            "actions":
                training.get(
                    "actions"
                ),

            "fallback_action":
                training.get(
                    "fallback_action"
                ),

            "reward":
                training.get(
                    "reward"
                ),

            "split":
                training.get(
                    "split"
                ),

            "best_epoch":
                training[
                    "selection"
                ][
                    "best_epoch"
                ],

            "best_validation_reward":
                training[
                    "selection"
                ][
                    "best_validation_reward"
                ],

            "best_model":
                training[
                    "selection"
                ][
                    "best_model"
                ],
        },

        "held_out_comparison": {
            "source":
                str(COMPARISON_REPORT),

            "validation_trajectories":
                comparison[
                    "validation_trajectories"
                ],

            "validation_states":
                comparison[
                    "validation_states"
                ],

            "target_mass":
                comparison[
                    "target_mass"
                ],

            "reward_definition":
                comparison[
                    "reward_definition"
                ],

            "results":
                results,
        },

        "headline": {
            "best_controller_by_common_reward":
                best_controller,

            "adaptive_topk_beats_dqn":
                adaptive_beats_dqn,

            "DQN": {
                "mean_reward":
                    dqn["mean_reward"],

                "mean_K":
                    dqn["mean_K"],

                "mean_covered_mass":
                    dqn[
                        "mean_covered_mass"
                    ],

                "outage_rate":
                    dqn[
                        "outage_rate"
                    ],

                "mean_switching_cost":
                    dqn[
                        "mean_switching_cost"
                    ],

                "fallback_rate":
                    dqn[
                        "fallback_rate"
                    ],
            },

            "AdaptiveTopK95": {
                "mean_reward":
                    adaptive[
                        "mean_reward"
                    ],

                "mean_K":
                    adaptive[
                        "mean_K"
                    ],

                "mean_covered_mass":
                    adaptive[
                        "mean_covered_mass"
                    ],

                "outage_rate":
                    adaptive[
                        "outage_rate"
                    ],

                "mean_switching_cost":
                    adaptive[
                        "mean_switching_cost"
                    ],

                "fallback_rate":
                    adaptive[
                        "fallback_rate"
                    ],
            },

            "FixedK1": {
                "mean_reward":
                    fixed1[
                        "mean_reward"
                    ],

                "outage_rate":
                    fixed1[
                        "outage_rate"
                    ],
            },

            "FixedK3": {
                "mean_reward":
                    fixed3[
                        "mean_reward"
                    ],

                "outage_rate":
                    fixed3[
                        "outage_rate"
                    ],
            },

            "FixedK5": {
                "mean_reward":
                    fixed5[
                        "mean_reward"
                    ],

                "outage_rate":
                    fixed5[
                        "outage_rate"
                    ],
            },
        },

        "conclusion": (
            "Under the common held-out reward and action/fallback "
            "definition, Adaptive Top-K at 95% posterior mass "
            "outperformed the learned DQN policy. The DQN achieved "
            "slightly higher mean posterior coverage but required "
            "more beams, incurred higher switching cost, and retained "
            "a non-zero outage rate. Therefore the deterministic "
            "Adaptive Top-K controller remains the preferred Stage-5 "
            "beam-selection policy, while DQN is retained as an "
            "exploratory learned-policy extension."
        ),

        "generated_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # Print final summary
    # --------------------------------------------------------

    print("=" * 78)
    print("FINAL REAL RL RESULTS")
    print("=" * 78)

    print()
    print(
        "Best controller:",
        best_controller
    )

    print()
    print("Adaptive Top-K 95%")
    print(
        "  reward =",
        adaptive[
            "mean_reward"
        ]
    )
    print(
        "  mean K =",
        adaptive[
            "mean_K"
        ]
    )
    print(
        "  mass =",
        adaptive[
            "mean_covered_mass"
        ]
    )
    print(
        "  outage =",
        adaptive[
            "outage_rate"
        ]
    )
    print(
        "  switching =",
        adaptive[
            "mean_switching_cost"
        ]
    )
    print(
        "  fallback =",
        adaptive[
            "fallback_rate"
        ]
    )

    print()
    print("DQN")
    print(
        "  reward =",
        dqn[
            "mean_reward"
        ]
    )
    print(
        "  mean K =",
        dqn[
            "mean_K"
        ]
    )
    print(
        "  mass =",
        dqn[
            "mean_covered_mass"
        ]
    )
    print(
        "  outage =",
        dqn[
            "outage_rate"
        ]
    )
    print(
        "  switching =",
        dqn[
            "mean_switching_cost"
        ]
    )
    print(
        "  fallback =",
        dqn[
            "fallback_rate"
        ]
    )

    print()
    print(
        "Scientific conclusion:"
    )
    print(
        report[
            "conclusion"
        ]
    )

    print()
    print(
        "Saved:",
        OUTPUT
    )


if __name__ == "__main__":
    main()

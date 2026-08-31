from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, isfinite, lgamma, log, log1p


FORMAL_EMPIRICAL_ALPHA = 0.05

REQUESTED_COVERAGE_LEVELS = (
    0.90,
    0.95,
    0.975,
    0.99,
)

NOMINAL_COVERAGE_Q = 0.95

FORMAL_HORIZONS_S = (
    0.1,
    0.3,
    0.5,
    1.0,
)

FORMAL_CODEBOOK_SIZES = (
    16,
    32,
    64,
)

FIXED_TOPK_BASELINES = (
    1,
    3,
    5,
)

FORMAL_COVERAGE_METHOD = (
    "one_sided_exact_binomial_lower_tail_"
    "consistency_with_requested_q"
)


def _validate_probability(
    value: float,
    name: str,
) -> float:
    result = float(value)

    if (
        not isfinite(result)
        or
        not 0.0 < result < 1.0
    ):
        raise ValueError(
            f"{name} must lie strictly in (0,1)."
        )

    return result


def binomial_lower_tail(
    hits: int,
    trials: int,
    probability: float,
) -> float:
    """
    Exact lower-tail probability

        P[X <= hits], X ~ Binomial(trials, probability)

    evaluated in log-space for numerical stability.
    """

    n = int(trials)
    h = int(hits)
    p = _validate_probability(
        probability,
        "probability",
    )

    if n < 0:
        raise ValueError(
            "trials must be >= 0."
        )

    if h < 0:
        return 0.0

    if h >= n:
        return 1.0

    logs = []

    for k in range(
        0,
        h + 1,
    ):
        value = (
            lgamma(n + 1)
            -
            lgamma(k + 1)
            -
            lgamma(n - k + 1)
            +
            k
            *
            log(p)
            +
            (n - k)
            *
            log1p(-p)
        )

        logs.append(
            value
        )

    maximum = max(
        logs
    )

    result = (
        exp(maximum)
        *
        sum(
            exp(
                value
                -
                maximum
            )
            for value in logs
        )
    )

    return min(
        1.0,
        max(
            0.0,
            float(result),
        ),
    )


def minimum_accepted_hits(
    trials: int,
    requested_q: float,
    alpha: float = FORMAL_EMPIRICAL_ALPHA,
) -> int:
    """
    Smallest integer number of hits for which the
    observed lower-tail Binomial probability is not
    below the frozen rejection level alpha.

    Depends ONLY on:
        trials,
        requested_q,
        frozen alpha.

    It never depends on observed Stage5 formal results.
    """

    n = int(
        trials
    )

    if n <= 0:
        raise ValueError(
            "trials must be positive."
        )

    q = _validate_probability(
        requested_q,
        "requested_q",
    )

    a = _validate_probability(
        alpha,
        "alpha",
    )

    for hits in range(
        0,
        n + 1,
    ):
        p_value = (
            binomial_lower_tail(
                hits,
                n,
                q,
            )
        )

        if (
            p_value
            >=
            a
        ):
            return hits

    raise RuntimeError(
        "Could not resolve minimum accepted hits."
    )


@dataclass(
    frozen=True
)
class FormalCoverageDecision:
    requested_q: float
    hits: int
    trials: int
    empirical_coverage: float
    lower_tail_p_value: float
    alpha: float
    minimum_accepted_hits: int
    minimum_accepted_empirical_coverage: float
    status: str

    @property
    def passed(
        self,
    ) -> bool:
        return (
            self.status
            ==
            "PASS_CONSISTENT_WITH_REQUESTED_Q"
        )

    def to_dict(
        self,
    ):
        payload = asdict(
            self
        )

        payload[
            "passed"
        ] = self.passed

        return payload


def empirical_coverage_decision(
    *,
    hits: int,
    trials: int,
    requested_q: float,
    alpha: float = FORMAL_EMPIRICAL_ALPHA,
) -> FormalCoverageDecision:
    n = int(
        trials
    )

    h = int(
        hits
    )

    if n <= 0:
        raise ValueError(
            "trials must be positive."
        )

    if (
        h < 0
        or
        h > n
    ):
        raise ValueError(
            "hits must lie in [0,trials]."
        )

    q = _validate_probability(
        requested_q,
        "requested_q",
    )

    a = _validate_probability(
        alpha,
        "alpha",
    )

    p_value = (
        binomial_lower_tail(
            h,
            n,
            q,
        )
    )

    minimum_hits = (
        minimum_accepted_hits(
            n,
            q,
            a,
        )
    )

    passed = bool(
        p_value
        >=
        a
    )

    return FormalCoverageDecision(
        requested_q=q,

        hits=h,

        trials=n,

        empirical_coverage=(
            h
            /
            n
        ),

        lower_tail_p_value=(
            p_value
        ),

        alpha=a,

        minimum_accepted_hits=(
            minimum_hits
        ),

        minimum_accepted_empirical_coverage=(
            minimum_hits
            /
            n
        ),

        status=(
            "PASS_CONSISTENT_WITH_REQUESTED_Q"
            if passed
            else
            "FAIL_SIGNIFICANTLY_BELOW_REQUESTED_Q"
        ),
    )


def formal_acceptance_policy_dict():
    return {
        "stage":
            5,

        "block":
            "5.8",

        "status":
            "FROZEN_PRE_FORMAL",

        "formal_empirical_coverage_tolerance": {
            "representation":
                "finite_sample_statistical_acceptance_rule",

            "fixed_absolute_percentage_point_tolerance":
                None,

            "method":
                FORMAL_COVERAGE_METHOD,

            "alpha":
                FORMAL_EMPIRICAL_ALPHA,

            "null_boundary":
                (
                    "coverage_probability = requested_q"
                ),

            "lower_tail_p_value":
                (
                    "P[X <= observed_hits | "
                    "X~Binomial(n, requested_q)]"
                ),

            "pass_rule":
                (
                    "lower_tail_p_value >= 0.05"
                ),

            "interpretation":
                (
                    "no statistically significant evidence "
                    "that empirical containment is below "
                    "the requested posterior-mass target"
                ),

            "important_limitation":
                (
                    "PASS is finite-sample consistency with "
                    "requested_q; it is not proof that the "
                    "unknown population coverage is >= q"
                ),
        },

        "coverage_targets": {
            "reported_q":
                list(
                    REQUESTED_COVERAGE_LEVELS
                ),

            "nominal_primary_q":
                NOMINAL_COVERAGE_Q,

            "primary_acceptance_scope":
                (
                    "nominal_q_0.95"
                ),

            "secondary_targets":
                [
                    0.90,
                    0.975,
                    0.99,
                ],
        },

        "evaluation_granularity": {
            "codebook_sizes":
                list(
                    FORMAL_CODEBOOK_SIZES
                ),

            "horizons_s":
                list(
                    FORMAL_HORIZONS_S
                ),

            "primary_tests":
                (
                    "separate test for every "
                    "codebook_size x horizon"
                ),

            "pool_horizons_for_primary_test":
                False,

            "reason":
                (
                    "avoid treating correlated horizons "
                    "from one selected receiver/scene as "
                    "independent Bernoulli observations"
                ),

            "primary_completion_rule":
                (
                    "all evaluable nominal-q codebook/horizon "
                    "strata must pass; n=0 is NOT_EVALUABLE "
                    "and cannot silently pass"
                ),
        },

        "receiver_denominator_contract": {
            "receiver_policy":
                "nearest_causal_vehicle_ahead",

            "one_primary_receiver_per_formal_scene":
                True,

            "tracks_to_predict_receiver_selection":
                False,

            "future_truth_receiver_selection":
                False,

            "no_eligible_receiver":
                (
                    "count explicitly in receiver availability; "
                    "do not substitute another actor"
                ),

            "selected_receiver_prediction_unavailable":
                (
                    "count explicitly; do not substitute "
                    "the second-best receiver"
                ),

            "invalid_future_endpoint":
                (
                    "exclude only from that horizon's "
                    "Bernoulli denominator and report "
                    "the unavailable endpoint count"
                ),

            "future_truth_role":
                "evaluator_only_after_controller_decision",
        },

        "adaptive_overhead_acceptance": {
            "nominal_q":
                NOMINAL_COVERAGE_Q,

            "hard_exhaustive_requirement":
                (
                    "adaptive mean physical probe count "
                    "< exhaustive beam count for each "
                    "codebook"
                ),

            "fixed_topk_baselines":
                list(
                    FIXED_TOPK_BASELINES
                ),

            "fixed_comparator_rule":
                (
                    "first determine which fixed Top-K "
                    "baselines independently pass the same "
                    "nominal coverage criterion across all "
                    "four horizons; among coverage-valid "
                    "fixed baselines use the smallest K as "
                    "the fair overhead comparator"
                ),

            "adaptive_vs_valid_fixed_requirement":
                (
                    "adaptive mean physical probe count "
                    "<= best coverage-valid fixed Top-K"
                ),

            "if_no_fixed_topk_is_coverage_valid":
                (
                    "report that fact explicitly; fixed "
                    "comparison is descriptive and the "
                    "hard overhead requirement remains "
                    "adaptive < exhaustive"
                ),

            "why_top1_is_not_unconditional_target":
                (
                    "a lower-overhead fixed policy that fails "
                    "the requested reliability target is not "
                    "a valid reliability-overhead comparator"
                ),
        },

        "formal_leakage_guards": {
            "policy_frozen_before_formal_metrics":
                True,

            "formal_results_used_to_choose_alpha":
                False,

            "formal_results_used_to_choose_q":
                False,

            "formal_results_used_to_choose_threshold":
                False,

            "post_hoc_tolerance_change_allowed":
                False,

            "training_on_formal":
                False,

            "recalibration_on_formal":
                False,

            "model_selection_on_formal":
                False,
        },
    }

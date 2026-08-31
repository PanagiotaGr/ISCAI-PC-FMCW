from __future__ import annotations


RECEIVER_POLICY_STATUS = (
    "FROZEN"
)

PRIMARY_POLICY = (
    "nearest_causal_vehicle_ahead"
)

CONNECTIVITY_SEMANTICS = (
    "constructed_hypothetical_connected_vehicle"
)

MINIMUM_FORWARD_H0_M = (
    0.0
)

#
# Intentionally no additional communication-receiver
# distance cutoff.
#
# The PDF does not mandate such a threshold.  Freezing a
# finite 50/100/150 m value from downstream performance
# would introduce an unnecessary tunable parameter.
#
# Distance remains an evaluation/slicing variable later.
#
MAX_PLANAR_RANGE_M = (
    None
)

MAX_RANGE_SEMANTICS = (
    "no_additional_receiver_eligibility_range_cutoff"
)

MAX_RANGE_RESOLUTION_BASIS = (
    "preregistered_structural_choice_not_performance_tuned"
)

MAX_RANGE_DEVELOPMENT_TUNING_PERFORMED = (
    False
)

MAX_RANGE_FORMAL_TUNING_PERFORMED = (
    False
)

VEHICLE_CLASS_REQUIRED = (
    True
)

CURRENT_CAUSAL_AVAILABILITY_REQUIRED = (
    True
)

ASSOCIATION_VALID_REQUIRED = (
    True
)

POSITIVE_FORWARD_H0_REQUIRED = (
    True
)

TRACKS_TO_PREDICT_ALLOWED = (
    False
)

FUTURE_TRUTH_ALLOWED = (
    False
)

PERFECT_TRACK_ID_ALLOWED = (
    False
)

ORACLE_CONNECTIVITY_ALLOWED = (
    False
)

NO_ELIGIBLE_RECEIVER_STATE = (
    "NO_ELIGIBLE_RECEIVER"
)


def receiver_policy_dict():
    return {
        "status":
            RECEIVER_POLICY_STATUS,

        "primary_policy":
            PRIMARY_POLICY,

        "connectivity_semantics":
            CONNECTIVITY_SEMANTICS,

        "minimum_forward_H0_m":
            MINIMUM_FORWARD_H0_M,

        "max_planar_range_m":
            MAX_PLANAR_RANGE_M,

        "max_range_semantics":
            MAX_RANGE_SEMANTICS,

        "max_range_resolution_basis":
            MAX_RANGE_RESOLUTION_BASIS,

        "max_range_development_tuning_performed":
            MAX_RANGE_DEVELOPMENT_TUNING_PERFORMED,

        "max_range_formal_tuning_performed":
            MAX_RANGE_FORMAL_TUNING_PERFORMED,

        "eligibility": {
            "vehicle_class_required":
                VEHICLE_CLASS_REQUIRED,

            "current_causal_availability_required":
                CURRENT_CAUSAL_AVAILABILITY_REQUIRED,

            "association_valid_required":
                ASSOCIATION_VALID_REQUIRED,

            "positive_forward_H0_required":
                POSITIVE_FORWARD_H0_REQUIRED,
        },

        "forbidden_inputs": {
            "tracks_to_predict":
                TRACKS_TO_PREDICT_ALLOWED,

            "future_truth":
                FUTURE_TRUTH_ALLOWED,

            "perfect_track_ID":
                PERFECT_TRACK_ID_ALLOWED,

            "oracle_connectivity":
                ORACLE_CONNECTIVITY_ALLOWED,
        },

        "no_eligible_receiver_state":
            NO_ELIGIBLE_RECEIVER_STATE,

        "distance":
            (
                "reported_and_sliced_later_"
                "but_not_used_as_extra_primary_"
                "eligibility_cutoff"
            ),
    }
